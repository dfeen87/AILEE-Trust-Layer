# Licensed under the PolyForm Noncommercial License 1.0.0
import os


def process_file(filepath, header):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return False

    if header.strip() in content:
        return False

    new_content = header + "\n" + content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Added header to: {filepath}")
    return True

def main():
    count = 0
    # Walk tests directory
    for root, dirs, files in os.walk('tests'):
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')

        for file in files:
            filepath = os.path.join(root, file)
            if file.endswith('.py'):
                header = "# Licensed under the PolyForm Noncommercial License 1.0.0"
                if process_file(filepath, header):
                    count += 1
            elif file.endswith('.rs'):
                header = "//! Licensed under the PolyForm Noncommercial License 1.0.0"
                if process_file(filepath, header):
                    count += 1

    print(f"Total test files modified: {count}")

if __name__ == '__main__':
    main()
