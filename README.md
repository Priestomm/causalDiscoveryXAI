# causalDiscoveryXAI

This project is part of the course on Explainable AI at the University of Verona. The main focus of this project is on causal discovery, which involves identifying causal relationships between variables in a given dataset. The goal is to develop methods and algorithms that can provide explanations for the observed data and help understand the underlying causal mechanisms.

## Getting Started

To get started with this project, follow the instructions below:

1. Clone the repository.
2. Install the required dependencies.
3. Download the datasets.
4. Run the main script to perform causal discovery on your dataset.

## Datasets

The project includes the following datasets:

- Boat: Contains a CSV file with normal and anomaly data related to boat operations.
- Pepper: Includes a CSV file with normal and anomaly data for the Pepper system.
- Swat: Consists of a CSV file with normal and anomaly data for the SWaT system.

## Usage
Run the main script (`main.py`) with the desired options:

    - `--mode`: Choose the analysis mode. Currently, only 'PCMCI' is supported.
    - `--dataset_name`: Select the dataset for analysis. Options are 'boat', 'pepper', or 'swat'.
    - `--assumption`: Choose the assumption for analysis. Options are 'linear' or 'not_linear'.

## Example Usage

```bash
python main.py --mode PCMCI --dataset_name swat --assumption linear
```

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under priestomm and sebadarconso



