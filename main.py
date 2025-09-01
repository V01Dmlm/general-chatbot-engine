import asyncio
import json
import random
from engine.chatbot_orchestrator import ChatbotEngine
from config import ENGINE_TYPE, LOG_FILE

# ----------------- Async input wrapper -----------------
async def async_input(prompt: str = "") -> str:
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, lambda: input(prompt))
    except EOFError:
        return "quit"

# ----------------- Typewriter-style output -----------------
async def type_out(text: str, base_delay: float = 0.03, punctuation_pause: float = 0.25):
    for char in text:
        print(char, end="", flush=True)
        await asyncio.sleep(base_delay + (punctuation_pause if char in ".!?" else random.uniform(-0.01, 0.02)))

    # blinking cursor
    for _ in range(3):
        print("_", end="\r", flush=True)
        await asyncio.sleep(0.4)
        print(" ", end="\r", flush=True)
        await asyncio.sleep(0.4)
    print()

# ----------------- Async log saving -----------------
async def save_logs(bot: ChatbotEngine, background=False):
    try:
        log_data = bot.context_manager.get_history()
        if background:
            await asyncio.sleep(0)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[!] Failed to save logs: {e}")

# ----------------- Main chat loop -----------------
async def chat_loop():
    bot = ChatbotEngine()
    print(f"🔥 {ENGINE_TYPE} Chatbot is ready! Type 'quit' to exit.")

    while True:
        user_input = await async_input("You: ")
        if not user_input.strip():
            continue  # ignore empty input
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting... saving logs.")
            await save_logs(bot)
            break

        # ----------------- Process input -----------------
        processed = bot.input_processor.process(user_input)
        cleaned_input = processed["cleaned"]

        # ----------------- Generate response with retries -----------------
        response = ""
        for attempt in range(3):
            try:
                response, _ = bot.get_response(cleaned_input)
                break
            except Exception as e:
                response = f"[!] Error generating response (attempt {attempt+1}): {e}"
                await asyncio.sleep(0.1)
        else:
            response = "[!] GPT2 failed to respond after 3 attempts."

        # ----------------- Post-process response -----------------
        await type_out(f"Bot: {response}")

        # Fire-and-forget background log saving
        asyncio.create_task(save_logs(bot, background=True))

# ----------------- Entrypoint -----------------
if __name__ == "__main__":
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Saving logs and exiting gracefully.")
    except Exception as e:
        print(f"[!] Fatal error: {e}")
