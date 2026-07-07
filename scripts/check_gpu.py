import torch


def main():
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print("device count:", device_count)

        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"device {i}: {props.name}")
            print(f"  capability: {props.major}.{props.minor}")
            print(f"  total memory: {props.total_memory / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()