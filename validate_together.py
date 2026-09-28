import sys

from validate_endpoint import main as validate_endpoint_main


def main():
    sys.argv = [
        sys.argv[0],
        "--base-url",
        "https://api.together.xyz",
        "--api-type",
        "chat",
        "--api-key-env",
        "TOGETHER_API_KEY",
        "--model",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        *sys.argv[1:],
    ]
    return validate_endpoint_main()


if __name__ == "__main__":
    raise SystemExit(main())
