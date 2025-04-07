from genalm_task2_3_classify_order import main

if __name__ == "__main__":
    import sys
    sys.argv.extend(["--label_type", "family", "--output_dir", "/work/sgk270/genalm_task2_new/family_output"])
    main()
