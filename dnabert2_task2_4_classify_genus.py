from dnabert2_task2_4_classify_order import main

if __name__ == "__main__":
    import sys
    sys.argv.extend(["--label_type", "genus", "--output_dir", "genus_output"])
    main()
