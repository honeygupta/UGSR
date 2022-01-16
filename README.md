# Unaligned Guided Thermal Image Super-Resolution
#### Authors: Honey Gupta and Kaushik Mitra, IIT Madras 
For queries, contact: `hn.gpt1@gmail.com`


###Installation

Required packages: 

    pip install -r requirements.txt

### Preparing the data

Create a test.csv file that contains a list of the input images and the corresponding guide images and place it in the `input` folder for inside the method directories. 
You can use the `create_dataset.py` script to create the csv files.

A sample `.csv` file can be found in the `input` folder for the images stored in `datasets` folder.

### Code Usage

Sample testing scripts:

    python main.py --checkpoint_dir=checkpoint/flir --log_dir=output/flir_test --config_filename=../configs/test.json
    python main.py --checkpoint_dir=checkpoint/cats --log_dir=output/cats_test --config_filename=../configs/test.json
    
*Note: Make sure to modify the `configs/test.json` with the appropriate dataset*

The results will be stored in `--log_dir` folder in the form of a html file. The `imgs` folder will contain all the raw inputs and outputs.


#### *More details on the training and testing procedure coming soon!* 

## Cite Us
```
@article{gupta2021toward,
  title={Toward Unaligned Guided Thermal Super-Resolution},
  author={Gupta, Honey and Mitra, Kaushik},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={433--445},
  year={2021},
  publisher={IEEE}
}
```

