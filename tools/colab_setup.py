from pathlib import Path


def main() -> None:
    base = Path('/content/pothole-dfine')
    dirs = [
        base,
        base / 'third_party',
        base / 'outputs',
        Path('/content/dataset'),
        Path('/content/dataset/images'),
        Path('/content/dataset/labels'),
        Path('/content/dataset/annotations'),
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print('Colab klasor yapisi hazirlandi:')
    for d in dirs:
        print(f' - {d}')


if __name__ == '__main__':
    main()
