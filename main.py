from engine.chatbot_orchestrator import ChatbotEngine

if __name__ == '__main__':
    bot = ChatbotEngine()
    print('Chatbot engine ready! Type "exit" to quit.')
    conversation_history = []
    while True:
        user_input = input('You: ')
        if user_input.lower() == 'exit':
            break
        response, conversation_history = bot.get_response(user_input, conversation_history)
        print(f'Bot: {response}')