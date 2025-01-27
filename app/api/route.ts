import runDeepSeek from '@/app/api/llm'
export async function GET() {
  await runDeepSeek()
  return Response.json({
    message: "Hello World"
  })
}

