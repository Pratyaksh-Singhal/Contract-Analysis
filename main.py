import sys
import os


class Color:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    DIM    = "\033[2m"

def green(t):  return f"{Color.GREEN}{t}{Color.RESET}"
def yellow(t): return f"{Color.YELLOW}{t}{Color.RESET}"
def red(t):    return f"{Color.RED}{t}{Color.RESET}"
def cyan(t):   return f"{Color.CYAN}{t}{Color.RESET}"
def bold(t):   return f"{Color.BOLD}{t}{Color.RESET}"
def dim(t):    return f"{Color.DIM}{t}{Color.RESET}"


BANNER = f"""
{Color.CYAN}{Color.BOLD}
╔══════════════════════════════════════════╗
║     AI Contract Analysis System          ║
║     Powered by Gemini + ChromaDB         ║
╚══════════════════════════════════════════╝
{Color.RESET}"""

HELP = f"""
{bold('Usage:')}
  python main.py ingest                  Ingest all files from contracts/ folder
  python main.py ingest <file>           Ingest a single file
  python main.py query                   Start interactive Q&A session
  python main.py query "<question>"      Ask one question and exit
  python main.py status                  Show system status
  python main.py clear                   Clear all ingested documents

{bold('Supported formats:')} PDF, TXT, DOCX

{bold('Quick start:')}
  1. Add contracts to the contracts/ folder
  2. python main.py ingest
  3. python main.py query
"""


def build_components():
    from src.config import config
    from src.vector_store import ChromaVectorStore
    from src.llm import create_llm
    from src.ingestion import IngestionPipeline
    from src.query_engine import QueryEngine

    vector_store = ChromaVectorStore(config.vector_store, config.embedding)
    llm          = create_llm(config.llm)
    ingestion    = IngestionPipeline(config, vector_store)
    engine       = QueryEngine(vector_store, llm, config.llm)

    return config, vector_store, llm, ingestion, engine


def cmd_ingest(args):
    config, vector_store, _, ingestion, _ = build_components()

    if args:
        target = args[0]
        if not os.path.exists(target):
            print(red(f"Error: file not found — {target}"))
            sys.exit(1)
        print(f"\nIngesting: {bold(target)}")
        count = ingestion.ingest_file(target)
        print(green(f"\n✓ Done — {count} chunks stored."))
    else:
        contracts_dir = config.contracts_dir
        os.makedirs(contracts_dir, exist_ok=True)

        files = [f for f in os.listdir(contracts_dir)
                 if os.path.isfile(os.path.join(contracts_dir, f)) and not f.startswith('.')]

        if not files:
            print(yellow(f"\nNo files found in {contracts_dir}/"))
            print(dim("Drop your PDF, TXT, or DOCX contracts there and run again."))
            return

        print(f"\nIngesting from: {bold(contracts_dir)}/")
        print(dim(f"Found {len(files)} file(s)...\n"))

        total_chunks, processed = ingestion.ingest_directory(contracts_dir)

        if processed:
            print(green(f"\n✓ Done — {total_chunks} chunks stored from {len(processed)} file(s)."))
        else:
            print(yellow("\nNo supported files were processed."))


def cmd_query(args):
    config, vector_store, llm, _, engine = build_components()

    if not llm.is_available():
        print(yellow("\n⚠  Warning: LLM not available."))
        if config.llm.provider == "gemini":
            print(dim("   Set your API key: export GEMINI_API_KEY=your_key_here"))
            print(dim("   Get a free key at: https://aistudio.google.com/app/apikey"))
        else:
            print(dim("   Start Ollama: ollama serve"))

    if args:
        question = " ".join(args)
        print(dim("\nQuerying..."))
        result = engine.query(question)
        print(engine.format_result(result))
        return

    print(f"\n{bold('Interactive Q&A')} {dim('(type exit or quit to stop)')}\n")

    while True:
        try:
            question = input(cyan("You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print(dim("\nExiting."))
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print(dim("Goodbye."))
            break

        print(dim("Thinking..."))
        result = engine.query(question)
        print(engine.format_result(result))


def cmd_status(args):
    config, vector_store, llm, _, _ = build_components()

    print(f"\n{bold('System Status')}")
    print("─" * 40)

    count = vector_store.document_count()
    print(f"  Vector store : {green(f'{count} chunks') if count > 0 else yellow('0 chunks (empty)')}")

    if llm.is_available():
        provider_info = f"{config.llm.provider} / {config.llm.model_name}"
        print(f"  LLM          : {green('ready')} — {dim(provider_info)}")
    else:
        print(f"  LLM          : {red('not ready')}")
        if config.llm.provider == "gemini":
            print(f"                 set GEMINI_API_KEY environment variable")
        else:
            print(f"                 run `ollama serve`")

    print(f"  Embedding    : {dim(config.embedding.model_name)}")
    print(f"  Contracts dir: {dim(config.contracts_dir)}")
    print("─" * 40 + "\n")


def cmd_clear(args):
    _, vector_store, _, _, _ = build_components()
    confirm = input(yellow("Delete all ingested documents? (y/N): ")).strip().lower()
    if confirm == "y":
        vector_store.clear()
        print(green("✓ Cleared."))
    else:
        print(dim("Cancelled."))


COMMANDS = {
    "ingest": cmd_ingest,
    "query":  cmd_query,
    "status": cmd_status,
    "clear":  cmd_clear,
}

if __name__ == "__main__":
    print(BANNER)

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        sys.exit(0)

    command = sys.argv[1].lower()
    rest    = sys.argv[2:]

    if command not in COMMANDS:
        print(red(f"Unknown command: {command}"))
        print(HELP)
        sys.exit(1)

    try:
        COMMANDS[command](rest)
    except KeyboardInterrupt:
        print(dim("\nInterrupted."))
    except Exception as e:
        print(red(f"\nError: {e}"))
        sys.exit(1)
