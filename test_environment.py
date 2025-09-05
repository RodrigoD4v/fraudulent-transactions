import sys

REQUIRED_VERSION = (3, 9, 19)

def main():
    system_version = sys.version_info[:3]

    if system_version != REQUIRED_VERSION:
        print(
            f">>>ERRO: Este projeto requer Python {'.'.join(map(str, REQUIRED_VERSION))}. "
            f"Encontrado: Python {'.'.join(map(str, system_version))}"
        )
        sys.exit(1)
    else:
        print(">>> Ambiente de desenvolvimento está configurado corretamente com Python 3.9.19!")

if __name__ == "__main__":
    main()
