from PIL import Image, UnidentifiedImageError
import sys
import os

if __name__ == "__main__":
    argv = sys.argv
    print("Args: " + str(argv))
    if (len(argv) < 2 or len(argv) > 3):
        print("Usage: " + argv[0] + ": `" + os.path.basename(argv[0]) + " <directory from> [<directory to>]'",
              file=sys.stderr)
        exit(1)

    entries = None
    try:
        entries = os.listdir(path=argv[1])
    except FileNotFoundError as FNFE:
        print(type(FNFE).__name__ + ": " + argv[0] + ": " + str(FNFE))
        exit(2)

    print("Directory [" + argv[1] + "]: " + str(entries))

    for item in entries:
        error = False
        fin = None
        if (item == "Grey-Scale"):
            continue
        try:
            filename = os.path.join(argv[1], item)
            fin = Image.open(filename)
        except PermissionError as PE:
            print(type(PE).__name__ + ": " + argv[0] + ": item `" + filename + "' is not a file")
            continue
        except UnidentifiedImageError as UIE:
            print(type(UIE).__name__ + ": " + argv[0] + ": item `" + filename + "' is not an image")
            continue
        fin = fin.convert('L')
        if (len(argv) == 3):
            try:
                filename = os.path.join(argv[2], item)
                fin.save(filename)
                print("Converted and saved [" + item + "] to directory [" + argv[2] + "]")
            except FileNotFoundError as FNFE:
                print(type(FNFE).__name__ + ": " + argv[0] + ": invalid path `" + filename +
                      "'\n\t- will attempt to save within [" + argv[1] + "]")
                error = True

        if(len(argv) == 2 or error == True):
            dir_path = os.path.join(argv[1], "Gray-Scale")
            path = None
            try:
                os.mkdir(dir_path)
            except FileExistsError:
                pass
            try:
                path = os.path.join(dir_path, item)
                fin.save(path)
            except FileNotFoundError as FNFE:
                print(type(FNFE).__name__ + ": " + argv[0] + ": invalid path `" + path + "'")
                exit(2)
            print("Converted and saved [" + item + "] to directory [" + dir_path + "]")

    print("Done")
