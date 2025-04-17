// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview
client<llm> Flash20 {
  provider google-ai
  options {
    model "gemini-2.0-flash"
    api_key env.OPENAI_API_KEY
  }
}

client<llm> Pro25 {
  provider google-ai
  options {
    model "gemini-2.0-flash"
    api_key env.OPENAI_API_KEY
  }
}
