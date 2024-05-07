import argparse

import torch

from CNNModel_OverFit import SimpleCNN
from Simulation import run_simulation_for_fixed_ship, run_simulation_for_learned_bot, \
    run_simulation_for_learned_bot_for_k_positions_of_bot_and_crew
from DataGenerator import DataGenerator


def main():
    # # run_simulation_for_fixed_ship()
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('-v', '--version', action='version', version='1.0')
    # subparsers = parser.add_subparsers(dest='command', required=True)
    # data_generator = subparsers.add_parser('generate_data_for_generalizing',
    #                                        help='Run simulation with Bot1')
    # data_generator.add_argument('-n', '--num_ship', required=True, type=int,
    #                             help='The number of ships to generate the dataset. The same number of files will be '
    #                                  'generated')
    # args = parser.parse_args()
    # if args.command == 'generate_data_for_generalizing':
    #     data_generator = DataGenerator(args.num_ship)
    #     data_generator.generate_data()

    model_path = '/common/home/kd958/PycharmProjects/Robot-Guidance/best-CNN-Overfit_new.pt'
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model = model.float()  # Ensure the model is using float32
    model.eval()  # Switch the model to evaluation mode
    run_simulation_for_learned_bot_for_k_positions_of_bot_and_crew(model, 100)


if __name__ == '__main__':
    main()
