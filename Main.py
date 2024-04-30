import argparse

from Simulation import run_simulation_for_fixed_ship
from DataGenerator import DataGenerator

def main():
    # run_simulation_for_fixed_ship()
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version='1.0')
    subparsers = parser.add_subparsers(dest='command', required=True)
    data_generator = subparsers.add_parser('generate_data_for_generalizing',
                                      help='Run simulation with Bot1')
    data_generator.add_argument('-n','--num_ship', required=True, type=int,
                                help='The number of ships to generate the dataset. The same number of files will be '
                                       'generated')
    args = parser.parse_args()
    if args.command == 'generate_data_for_generalizing':
        data_generator = DataGenerator(args.num_ship)
        data_generator.generate_data()


if __name__ == '__main__':
    main()
