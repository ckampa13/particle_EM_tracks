import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--directory', help='Output directory')
    parser.add_argument('-f', '--filename', help='Output base filename')
    parser.add_argument('-N', '--number', help='Number of events')

    args = parser.parse_args()

    if args.directory is None:
        args.directory = "/home/ckampa/data/pickles/distortions/linear_gradient/"
    if args.filename is None:
        args.filename = "test_01"
    if args.number is None:
        args.number = 10

    print(f"Directory: {args.directory},\nFilename: {args.filename},\nNumber: {args.number}")

parser.
