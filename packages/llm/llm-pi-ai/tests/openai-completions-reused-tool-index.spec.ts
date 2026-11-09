import { afterEach, describe, expect, it } from 'vitest'
import type { AssistantMessageEvent, Model } from '@earendil-works/pi-ai'
import { stream as streamCompletions } from '@earendil-works/pi-ai/api/openai-completions'
import { normalizeContext } from '@earendil-works/pi-ai/utils/transcript'
import { closeMockServers, mockServer } from './mock-server.ts'

afterEach(closeMockServers)

/**
 * One OpenAI Chat Completions tool-call delta.
 * @param id - the tool call id, absent on continuation frames.
 * @param name - the function name, absent on continuation frames.
 * @param arguments - the argument fragment.
 * @param index - the stream index the gateway assigned to the call.
 */
function toolCallFrame(id: string | null, name: string | null, args: string, index: number): string {
  return JSON.stringify({ choices: [{ delta: { tool_calls: [{ index, id, type: 'function', function: { name, arguments: args } }] }, index: 0, finish_reason: null }] })
}

function model(baseUrl: string): Model<'openai-completions'> {
  return {
    id: 'm', name: 'm', api: 'openai-completions', provider: 'test', baseUrl, reasoning: false, input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 8192, maxTokens: 1024,
  }
}

const context = normalizeContext({ messages: [{ role: 'user', content: 'hi', timestamp: 0 }] })

/** One finished tool call: its id, the parse error when it failed, and the parsed arguments. */
interface FinishedToolCall {
  id: string
  argumentsError: string | undefined
  arguments: unknown
}

/**
 * Collect every finished tool call with its parse error.
 * @param events - the provider event stream.
 * @returns one entry per finished tool call.
 */
async function collectToolCalls(events: AsyncIterable<AssistantMessageEvent>): Promise<FinishedToolCall[]> {
  const calls: FinishedToolCall[] = []
  for await (const event of events) {
    if (event.type === 'toolcall_end') {
      calls.push({
        id: event.toolCall.id,
        argumentsError: event.toolCall.argumentsError,
        arguments: event.toolCall.arguments,
      })
    }
  }
  return calls
}

describe('openai-completions with a reused tool-call stream index', () => {
  it('starts a new call when a new named id arrives at a reused index after the previous arguments close', async () => {
    const server = await mockServer([{ events: [
      toolCallFrame('call_A', 'bash', '{"command": "echo AAA", "description": "A"}', 0),
      toolCallFrame('call_B', 'bash', '{"command": "echo BBB", "description": "B"}', 0),
      '{"choices":[{"delta":{},"index":0,"finish_reason":"tool_calls"}]}',
      '[DONE]',
    ] }])
    const calls = await collectToolCalls(streamCompletions(model(server.url), context, { apiKey: 'test-key' }))
    expect(calls).toHaveLength(2)
    expect(calls[0]?.argumentsError).toBeUndefined()
    expect(calls[1]?.argumentsError).toBeUndefined()
    expect(calls[0]?.arguments).toEqual({ command: 'echo AAA', description: 'A' })
    expect(calls[1]?.arguments).toEqual({ command: 'echo BBB', description: 'B' })
  })

  it('splits a reused index when the second call arrives after continuation frames of the first', async () => {
    const server = await mockServer([{ events: [
      toolCallFrame('call_A', 'bash', '', 0),
      toolCallFrame(null, null, '{"command": "echo AAA", "description": "A"}', 0),
      toolCallFrame('call_B', 'bash', '{"command": "echo BBB", "description": "B"}', 0),
      '{"choices":[{"delta":{},"index":0,"finish_reason":"tool_calls"}]}',
      '[DONE]',
    ] }])
    const calls = await collectToolCalls(streamCompletions(model(server.url), context, { apiKey: 'test-key' }))
    expect(calls).toHaveLength(2)
    expect(calls.every(call => call.argumentsError === undefined)).toBe(true)
    expect(calls[0]?.arguments).toEqual({ command: 'echo AAA', description: 'A' })
    expect(calls[1]?.arguments).toEqual({ command: 'echo BBB', description: 'B' })
  })

  it('keeps a compliant 0/1 index pair as two independent calls', async () => {
    const server = await mockServer([{ events: [
      toolCallFrame('call_A', 'bash', '{"command": "echo AAA", "description": "A"}', 0),
      toolCallFrame('call_B', 'bash', '{"command": "echo BBB", "description": "B"}', 1),
      '{"choices":[{"delta":{},"index":0,"finish_reason":"tool_calls"}]}',
      '[DONE]',
    ] }])
    const calls = await collectToolCalls(streamCompletions(model(server.url), context, { apiKey: 'test-key' }))
    expect(calls).toHaveLength(2)
    expect(calls.every(call => call.argumentsError === undefined)).toBe(true)
    expect(calls[0]?.arguments).toEqual({ command: 'echo AAA', description: 'A' })
    expect(calls[1]?.arguments).toEqual({ command: 'echo BBB', description: 'B' })
  })

  it('keeps the merge failure when the previous arguments are still open at the reused index', async () => {
    const server = await mockServer([{ events: [
      toolCallFrame('call_A', 'bash', '{"command": "echo AAA', 0),
      toolCallFrame('call_B', 'bash', '{"command": "echo BBB", "description": "B"}', 0),
      '{"choices":[{"delta":{},"index":0,"finish_reason":"tool_calls"}]}',
      '[DONE]',
    ] }])
    const calls = await collectToolCalls(streamCompletions(model(server.url), context, { apiKey: 'test-key' }))
    expect(calls).toHaveLength(1)
    expect(calls[0]?.argumentsError).toBeDefined()
  })

  it('keeps one call when the same id continues across frames', async () => {
    const server = await mockServer([{ events: [
      toolCallFrame('call_A', 'bash', '{"command": "echo ', 0),
      toolCallFrame(null, null, 'AAA", "description": "A"}', 0),
      '{"choices":[{"delta":{},"index":0,"finish_reason":"tool_calls"}]}',
      '[DONE]',
    ] }])
    const calls = await collectToolCalls(streamCompletions(model(server.url), context, { apiKey: 'test-key' }))
    expect(calls).toHaveLength(1)
    expect(calls[0]?.argumentsError).toBeUndefined()
    expect(calls[0]?.arguments).toEqual({ command: 'echo AAA', description: 'A' })
  })
})
