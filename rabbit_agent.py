"""
RABBIT-2B AUTONOMOUS EXECUTIVE AGENT
Main Entrypoint
"""
import sys
from rabbit.agent import RabbitExecutiveAgent

def main():
    agent = RabbitExecutiveAgent()
    print("\n" + "=" * 65)
    print("  RABBIT-2B AUTONOMOUS EXECUTIVE AGENT RUNTIME v2.0")
    print("  Commands: /status, /plan, /pause, /resume, /stop, exit")
    print("=" * 65)

    while True:
        try:
            user_input = input("\nRabbit [Objective / Command] > ").strip()
            if not user_input or user_input.lower() in ("exit", "quit"):
                break
            if user_input.startswith("/"):
                agent.handle_command(user_input)
            else:
                res = agent.run(user_input)
                print("\n" + "=" * 65)
                print(f"  FINAL RUN STATUS: {res.get('status')}")
                if "evidence" in res:
                    print(f"  Tasks Succeeded: {res.get('tasks_succeeded')}/{res.get('tasks_total')}")
                print("=" * 65)
        except KeyboardInterrupt:
            print("\n[Rabbit] Interrupted by user.")
            break

if __name__ == "__main__":
    main()
