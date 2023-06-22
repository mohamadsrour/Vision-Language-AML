import os
import logging
from parse_args import parse_arguments
from load_data import build_splits_baseline, build_splits_domain_disentangle, build_splits_clip_disentangle, build_splits_baseline_dg, build_splits_domain_disentangle_dg, build_splits_clip_disentangle_dg
from experiments.baseline import BaselineExperiment
from experiments.domain_disentangle import DomainDisentangleExperiment
from experiments.clip_disentangle import CLIPDisentangleExperiment
from experiments.domain_disentangle_dg import DomainDisentangleDGExperiment
from experiments.clip_disentangle_dg import CLIPDomainDisentangleDGExperiment
from datetime import timedelta
import time

def setup_experiment(opt):
    
    if opt['experiment'] == 'baseline':
        experiment = BaselineExperiment(opt)
        train_loader, validation_loader, test_loader = build_splits_baseline(opt)
        return experiment, train_loader, validation_loader, test_loader
        
    elif opt['experiment'] == 'domain_disentangle':
        experiment = DomainDisentangleExperiment(opt)
        train_loader, val_loader, test_loader = build_splits_domain_disentangle(opt)
        return experiment, train_loader, val_loader, test_loader

    elif opt['experiment'] == 'clip_disentangle':
        experiment = CLIPDisentangleExperiment(opt)
        train_loader, val_loader, test_loader = build_splits_clip_disentangle(opt)
        return experiment, train_loader, val_loader, test_loader
    
    elif opt['experiment'] == 'baseline_dg':
        experiment = BaselineExperiment(opt)
        train_loader, validation_loader, test_loader = build_splits_baseline_dg(opt)
        return experiment, train_loader, validation_loader, test_loader

    elif opt['experiment']  == 'domain_disentangle_dg':
        experiment = DomainDisentangleDGExperiment(opt)
        train_loader, val_loader, test_loader = build_splits_domain_disentangle_dg(opt)
        return experiment, train_loader, val_loader, test_loader

    elif opt['experiment']  == 'clip_disentangle_dg':
        experiment = CLIPDomainDisentangleDGExperiment(opt)
        train_loader, val_loader, test_loader = build_splits_clip_disentangle_dg(opt)
        return experiment, train_loader, val_loader, test_loader

    else:
        raise ValueError('Experiment not yet supported.')
    
    return experiment, train_loader, validation_loader, test_loader

def main(opt):
    start = time.time()
    experiment, train_loader, validation_loader, test_loader = setup_experiment(opt)

    if not opt['test']: # Skip training if '--test' flag is set
        iteration = 0
        best_accuracy = 0
        total_train_loss = 0

        # Restore last checkpoint
        if os.path.exists(f'{opt["output_path"]}/last_checkpoint.pth'):
            iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        else:
            logging.info(opt)

        # Train loop
        while iteration < opt['max_iterations']:
            for data in train_loader:

                total_train_loss += experiment.train_iteration(data)

                if iteration % opt['print_every'] == 0:
                    logging.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                
                if iteration % opt['validate_every'] == 0:
                    # Run validation
                    val_accuracy, val_loss = experiment.validate(validation_loader)
                    logging.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                    experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)

                iteration += 1
                if iteration > opt['max_iterations']:
                    break

    finish = time.time()
    elapsed = finish - start
    logging.info(f'[TIME]: {timedelta(seconds=elapsed)}')

    # Test
    experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
    
    test_accuracy, _ = experiment.validate(test_loader)
    logging.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')


    if opt["validate_source"]:
        src_val_accuracy, _ = experiment.validate(validation_loader)
        logging.info(f'[VAL] Accuracy with best: {(100 * src_val_accuracy):.2f}')


if __name__ == '__main__':

    opt = parse_arguments()

    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='a')

    main(opt)
