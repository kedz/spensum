import sys
import os
import argparse
import colorama

import math
import torch
import ntp
import spensum

from collections import defaultdict
import rouge_papier
import pandas as pd

def burn_in(model, train_data, weight=None, report_every=100):

    max_iters = model.burn_in_iters
    model.train()
    crits = []
    for module in model.submodules:
        if not module.ready:
            crit = ntp.criterion.BinaryCrossEntropy(
                mode="logit", weight=weight, mask_value=-1)
            crit.add_reporter(
                    ntp.criterion.BinaryFMeasureReporter(mode="logit"))
            crits.append(crit)

    modules = [m for m in model.submodules]
    opts = [ntp.optimizer.Adam(m.parameters(), lr=.0001) for m in modules
            if not m.ready]

    n_iter = 0

    def print_report():
        sys.stdout.write(" "  * len(msg))
        sys.stdout.write("\r")
        print("\nBurn In iter={}".format(n_iter))
        for module, crit in zip(modules, crits):
            heading = "\n" + module.name
            if module.ready:
                heading += colorama.Fore.GREEN + " READY" + \
                    colorama.Fore.RESET
            else:
                heading += colorama.Fore.YELLOW + " BURNIN" + \
                    colorama.Fore.RESET
            print(heading)
            print(crit.report())

    while n_iter < max_iters:
        for batch in train_data.iter_batch():
            n_iter += 1
            msg = "burn in: {}/{}\r".format(n_iter, max_iters)
            sys.stdout.write(msg)
            sys.stdout.flush()

            for i, (crit, module, opt) in enumerate(zip(crits, modules, opts)):
                if not module.ready:
                    crit.minimize(batch, module, opt)

            if n_iter % report_every == 0:
                print_report()

            if n_iter == max_iters:
                break
    
    print("")
    
def collect_reference_paths(reference_dir):
    ids2refs = defaultdict(list)
    for filename in os.listdir(reference_dir):
        id = filename.rsplit(".", 2)[0]
        ids2refs[id].append(os.path.join(reference_dir, filename))
    return ids2refs

def compute_rouge(model, dataset, reference_dir):

    model.eval()

    ids2refs = collect_reference_paths(reference_dir)

    with rouge_papier.util.TempFileManager() as manager:

        path_data = []
        for batch in dataset.iter_batch():
            texts = model.extract(batch.inputs, batch.metadata, 
                                  strategy="rank")

            for id, summary in zip(batch.metadata.id, texts):
                summary_path = manager.create_temp_file(summary)
                ref_paths = ids2refs[id]
                path_data.append([summary_path, ref_paths])

        config_text = rouge_papier.util.make_simple_config_text(path_data)
        config_path = manager.create_temp_file(config_text)
        df = rouge_papier.compute_rouge(config_path, max_ngram=2, lcs=False)
        return df[-1:]

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-inputs", type=str, required=True)
    parser.add_argument("--train-labels", type=str, required=True)
    parser.add_argument("--valid-inputs", type=str, required=True)
    parser.add_argument("--valid-labels", type=str, required=True)
    parser.add_argument("--valid-summary-dir", type=str, required=True)

    parser.add_argument(
        "--gpu", default=-1, type=int, required=False)
    parser.add_argument(
        "--epochs", default=100, type=int, required=False)
    parser.add_argument(
        "--seed", default=83432534, type=int, required=False)

    parser.add_argument(
        "--lr", required=False, default=.01, type=float)
    parser.add_argument(
        "--batch-size", default=16, type=int, required=False)
    parser.add_argument(
        "--embedding-size", type=int, required=False, default=300)

    parser.add_argument(
        "--rnn-salience", action="store_true", default=False)
    parser.add_argument(
        "--rs-xent", type=float, required=False, default=0)
    parser.add_argument(
        "--rs-burn-in", type=int, default=0, required=False)
    parser.add_argument(
        "--rs-hidden-size", default=150, type=int, required=False)


    parser.add_argument(
        "--position", action="store_true", default=False)
    parser.add_argument(
        "--p-xent", type=float, default=0, required=False)
    parser.add_argument(
        "--p-num-positions", type=int, default=50, required=False)
    parser.add_argument(
        "--p-burn-in", type=int, default=0, required=False)

    parser.add_argument(
        "--word-count", action="store_true", default=False)
    parser.add_argument(
        "--wc-xent", type=float, default=0, required=False)
    parser.add_argument(
        "--wc-burn-in", type=int, default=0, required=False)

    parser.add_argument(
        "--neighbor-clique", action="store_true", default=False)

    parser.add_argument(
        "--report-every", type=int, required=False, default=1000)
    parser.add_argument(
        "--validate-every", type=int, required=False, default=1)
    parser.add_argument(
        "--burn-in-report-every", type=int, required=False, default=500)

    parser.add_argument(
        "--pc-coverage", action="store_true", default=False)
    parser.add_argument(
        "--pcc-xent", type=float, required=False, default=0)
    parser.add_argument(
        "--pcc-burn-in", type=int, default=0, required=False)

    parser.add_argument(
        "--hidden-layer-sizes", nargs="+", default=[100], type=int,
        required=False)
    parser.add_argument(
        "--hidden-layer-activations", nargs="+", default="relu", type=str,
        required=False)
    parser.add_argument(
        "--hidden-layer-dropout", default=.0, type=float, required=False)
    parser.add_argument(
        "--input-layer-norm", default=False, action="store_true")

#    parser.add_argument(
#        "--save-module", required=True, type=str)
#    parser.add_argument(
#        "--save-predictor", default=None, required=False, type=str)

    args = parser.parse_args(args)


    colorama.init()
    print("")
    print(
        "   ++=============================================================++")
    print("   || Summary Energy Network Sentence Extractor " + \
        colorama.Fore.GREEN + colorama.Style.BRIGHT + "(SENSEi)" + \
        colorama.Fore.RESET + colorama.Style.NORMAL + " trainer. ||" )
    print(
        "   ++=============================================================++")
    print("")
    
    
    print("Setting random seed: " + colorama.Style.BRIGHT + \
            colorama.Fore.WHITE + str(args.seed) + \
            colorama.Style.NORMAL + colorama.Fore.RESET + "\n")
    ntp.set_random_seed(args.seed)
    
    input_reader = spensum.dataio.init_duc_sds_input_reader(
        args.embedding_size)
    label_reader = spensum.dataio.init_duc_sds_label_reader()

    train_dataset = spensum.dataio.read_input_label_dataset(
        args.train_inputs, args.train_labels, 
        input_reader, label_reader,
        batch_size=args.batch_size, gpu=args.gpu)

    valid_dataset = spensum.dataio.read_input_label_dataset(
        args.valid_inputs, args.valid_labels, 
        input_reader, label_reader,
        batch_size=args.batch_size, gpu=args.gpu)

    non_salient_count = train_dataset.targets.eq(0).sum()
    salient_count = train_dataset.targets.eq(1).sum()
    weight = torch.FloatTensor([1 / non_salient_count, 1 / salient_count])

    crit = spensum.criterion.SPENLoss(weight=weight)
    crit.add_reporter(
        ntp.criterion.BinaryFMeasureReporter(mode="prob"))
    crit.set_selection_criterion("BinaryFMeasureReporter")

    print(colorama.Style.BRIGHT + "Beginning preflight check...\n" \
            + colorama.Style.NORMAL)



    print("Initializing submodules...")


    submodules = []
    
    if args.rnn_salience:
        module = spensum.module.RNNSalience(
            args.embedding_size, hidden_size=args.rs_hidden_size,
            burn_in=args.rs_burn_in)
        submodules.append(module)
        msg = "    {:>15} ... {:>8}".format(
            "rnn salience",
            colorama.Fore.GREEN + 'OK' + colorama.Fore.RESET)
        if module.burn_in > 0:
            msg += "   burnin={} iters".format(module.burn_in)
        if args.rs_xent > 0:
            aux_crit = ntp.criterion.BinaryCrossEntropy(
                name="RNNSalienceXEnt",
                mode="logit", weight=weight, mask_value=-1)
            aux_crit.add_reporter(
                ntp.criterion.BinaryFMeasureReporter(mode="logit"))
            crit.add_aux_criterion(module, aux_crit, args.rs_xent)
            msg += colorama.Fore.GREEN + "   xent obj" + colorama.Fore.RESET
        print(msg)
    else:    
        print("    {:>15} ... {:>8}".format(
            "rnn salience",
            colorama.Fore.YELLOW + 'SKIP' + colorama.Fore.RESET))

    if args.position:
        module = spensum.module.Position(
            args.p_num_positions, burn_in=args.p_burn_in)
        submodules.append(module)
        msg = "    {:>15} ... {:>8}".format(
            "position",
            colorama.Fore.GREEN + 'OK' + colorama.Fore.RESET)
        if module.burn_in > 0:
            msg += "   burnin={} iters".format(module.burn_in)
        if args.p_xent > 0:
            aux_crit = ntp.criterion.BinaryCrossEntropy(
                name="PositionXEnt",
                mode="logit", weight=weight, mask_value=-1)
            aux_crit.add_reporter(
                ntp.criterion.BinaryFMeasureReporter(mode="logit"))
            crit.add_aux_criterion(module, aux_crit, weight=args.p_xent)
            msg += colorama.Fore.GREEN + "   xent obj" + colorama.Fore.RESET
        print(msg)
    else:    
        print("    {:>15} ... {:>8}".format(
            "position",
            colorama.Fore.YELLOW + 'SKIP' + colorama.Fore.RESET))

    if args.word_count:
        module = spensum.module.WordCount(burn_in=args.wc_burn_in)
        submodules.append(module)
        msg = "    {:>15} ... {:>8}".format(
            "word_count",
            colorama.Fore.GREEN + 'OK' + colorama.Fore.RESET)
        if module.burn_in > 0:
            msg += "   burnin={} iters".format(module.burn_in)
        if args.wc_xent > 0:
            aux_crit = ntp.criterion.BinaryCrossEntropy(
                name="WordCountXEnt",
                mode="logit", weight=weight, mask_value=-1)
            aux_crit.add_reporter(
                ntp.criterion.BinaryFMeasureReporter(mode="logit"))
            crit.add_aux_criterion(module, aux_crit, weight=args.wc_xent)
            msg += colorama.Fore.GREEN + "   xent obj" + colorama.Fore.RESET
        print(msg)
    else:    
        print("    {:>15} ... {:>8}".format(
            "word_count",
            colorama.Fore.YELLOW + 'SKIP' + colorama.Fore.RESET))

    if args.pc_coverage:
        module = spensum.module.PCCoverage(
            args.embedding_size, burn_in=args.pcc_burn_in)
        submodules.append(module)
        msg = "    {:>15} ... {:>8}".format(
            "pc_coverage",
            colorama.Fore.GREEN + 'READY' + colorama.Fore.RESET)
        if module.burn_in > 0:
            msg += "   burnin={} iters".format(module.burn_in)
        if args.pcc_xent > 0:
            aux_crit = ntp.criterion.BinaryCrossEntropy(
                name="PCCoverageXEnt",
                mode="logit", weight=weight, mask_value=-1)
            aux_crit.add_reporter(
                ntp.criterion.BinaryFMeasureReporter(mode="logit"))
            crit.add_aux_criterion(module, aux_crit, weight=args.pcc_xent)
            msg += colorama.Fore.GREEN + "   xent obj" + colorama.Fore.RESET
        print(msg)
    else:    
        print("    {:>15} ... {:>8}".format(
            "pc_coverage",
            colorama.Fore.YELLOW + 'SKIP' + colorama.Fore.RESET))

    if args.neighbor_clique:
        module = spensum.module.NeighborClique()
        submodules.append(module)
        msg = "    {:>15} ... {:>8}".format(
            "neighbor_clique",
            colorama.Fore.GREEN + 'READY' + colorama.Fore.RESET)
        if module.burn_in > 0:
            msg += "   burnin={} iters".format(module.burn_in)
#        if args.nc_xent > 0:
#            aux_crit = ntp.criterion.BinaryCrossEntropy(
#                name="NCliqueXEnt",
#                mode="logit", weight=weight, mask_value=-1)
#            aux_crit.add_reporter(
#                ntp.criterion.BinaryFMeasureReporter(mode="logit"))
#            crit.add_aux_criterion(module, aux_crit, weight=args.pcc_xent)
#            msg += colorama.Fore.GREEN + "   xent obj" + colorama.Fore.RESET
        print(msg)
    else:    
        print("    {:>15} ... {:>8}".format(
            "neighbor_clique",
            colorama.Fore.YELLOW + 'SKIP' + colorama.Fore.RESET))






    print("\nInitializing energy model...")
    model = spensum.model.EnergyModel(submodules)
    if args.gpu > -1:
        print("Placing model on gpu device: " + \
            colorama.Style.BRIGHT + colorama.Fore.WHITE + str(args.gpu) \
            + colorama.Style.NORMAL + colorama.Fore.RESET)
        model.cuda(args.gpu)


    
   






    if not model.ready:
        print("Running burn in for {} iters...".format(model.burn_in_iters))
        burn_in(
            model, train_dataset, weight=weight, 
            report_every=args.burn_in_report_every)
 
    opt = ntp.optimizer.Adam(model.parameters(), lr=args.lr)

    max_iters = 1000000
    fit_model(model, crit, opt, train_dataset, max_iters,
              validation_dataset=valid_dataset, 
              report_every=args.report_every,
              validate_every=args.validate_every,
              validation_summary_dir=args.valid_summary_dir)


def fit_model(model, crit, opt, train_dataset, max_iters,
              report_every=500, validate_every=1, 
              validation_dataset=None, validation_summary_dir=None):


    def validate(report_iter):
        if validation_dataset is None:
            return
        elif report_iter % validate_every == 0:
            model.eval()
            max_valid_iters = math.ceil(
                validation_dataset.size / validation_dataset.batch_size)

            print(colorama.Style.BRIGHT + colorama.Fore.GREEN + \
                "  Validating..." + colorama.Style.NORMAL + \
                colorama.Fore.RESET)

            valid_iter = 0
            for batch in validation_dataset.iter_batch():
                valid_iter += 1
                crit.compute_loss(batch, model, opt)

                sys.stdout.write(
                    " {}/{} margin={:0.9f} aux={:0.9f}\r".format(
                        valid_iter, max_valid_iters, crit.margin_loss, 
                        crit.aux_loss))
                sys.stdout.flush()
            print("\n")
            print(crit.report(indent="  "))
            crit.reset()
            
            if validation_summary_dir is not None:
                df = compute_rouge(
                    model, validation_dataset, validation_summary_dir)
                df_lines = str(df).split("\n")
                print("   " + df_lines[0] + "\n   " + df_lines[2] + "\n")


            model.train()





    n_iter = 0
    report_iter = 0
    while n_iter < max_iters:
        
        for batch in train_dataset.iter_batch():
            n_iter += 1
            crit.minimize(batch, model, opt)

            sys.stdout.write(
                "training {}/{} margin={:0.9f} aux={:0.9f}\r".format(
                    n_iter, max_iters, crit.margin_loss, crit.aux_loss))
            sys.stdout.flush()

            if n_iter % report_every == 0:
                report_iter += 1
                print("\n")
                #print(crit.report(indent="  "))
                crit.reset()
                validate(report_iter)
            
            if n_iter >= max_iters:
                break


    exit()
    crit = SquaredLossAugmentedHingeLoss(weight=weight)
    #crit = ntp.criterion.BinaryCrossEntropy(
    #    mode="prob", weight=weight, mask_value=-1)
    crit.add_reporter(
        ntp.criterion.BinaryFMeasureReporter(mode="prob"))
    crit.add_reporter(
        PredictionHistogramReporter())
    crit.set_selection_criterion("BinaryFMeasureReporter")
    crit.use_margin = False



    train_rouge_results = []
    valid_rouge_results = []
    best_rouge = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == 10:
            crit.use_margin = True


        def train_step_callback(step, max_steps, batch_loss, criterion):
            sys.stdout.write("\r")
            sys.stdout.write(" " * 79)
            sys.stdout.write("\r")
            sys.stdout.write(
                "\ttrain {}: {} / {} | obj: {:0.9f}".format(
                    epoch, step, max_steps, criterion.avg_loss))
            sys.stdout.flush()
            if step == max_steps:
                sys.stdout.write("\r" + " " * 79 + "\r")
                sys.stdout.flush()

        def valid_step_callback(step, max_steps, batch_loss, criterion):
            sys.stdout.write("\r")
            sys.stdout.write(" " * 79)
            sys.stdout.write("\r")
            sys.stdout.write(
                "\tvalid {}: {} / {} | obj: {:0.9f}".format(
                    epoch, step, max_steps, criterion.avg_loss))
            sys.stdout.flush()
            if step == max_steps:
                sys.stdout.write("\r" + " " * 79 + "\r")
                sys.stdout.flush()


        ntp.trainer.train_epoch(
            crit, model, opt, train_dataset, step_callback=train_step_callback)
        crit.checkpoint("training")

        print(crit.report(indent="     "))
        #print(compute_rouge(model, train_dataset, args.train_summary_dir))

        #print("\n  * == Validation ==")

        ntp.trainer.eval(
            crit, model, valid_dataset, step_callback=valid_step_callback)
        crit.checkpoint("validation")
        
        print(crit.report(indent="     "))

        #best_epoch, obj = crit.find_best_checkpoint("validation")
        #if best_epoch == epoch and save_model is not None:
        #    torch.save(model, save_model)
        #print(crit.report(indent="     "))
        #print("\n     Best epoch: {} obj: {}\n".format(
        #    best_epoch, obj))
        #print("")

        valid_rouge = compute_rouge(
            model, valid_dataset, args.valid_summary_dir)
        print(epoch)
        print(valid_rouge)
        print("")
        valid_rouge_results.append(valid_rouge)
#?
#?        rouge_score = valid_rouge["rouge-2"].values[0]
#?        if rouge_score > best_rouge:
#?            best_rouge = rouge_score
#?            if args.save_model is not None:
#?                print("Saving model!")
#?                torch.save(model, args.save_model)


    #,
     #                              save_model=module_save_path)
    
    return pd.concat(valid_rouge_results, axis=0) 
    
    
    #exit()

    #train_data, valid_data = spensum.dataio.read_train_and_validation_data(
    #    args.train, args.valid, file_reader, args.batch_size, gpu=args.gpu)

#if __name__ == "__main__": 
#    main()
#
#
#    random.seed(args.seed)
#    torch.manual_seed(args.seed)
#    if args.gpu > -1:
#        torch.cuda.manual_seed(args.seed)
#
#    module = spensum.module.Salience(
#        args.embedding_size,
#        hidden_layer_sizes=args.hidden_layer_sizes,
#        hidden_layer_activations=args.hidden_layer_activations,
#        hidden_layer_dropout=args.hidden_layer_dropout,
#        input_layer_norm=args.input_layer_norm,
#        mode="pretrain")
#    if args.gpu > -1:
#        module.cuda(args.gpu)
#
#    file_reader = spensum.dataio.initialize_sds_reader(args.embedding_size)
#
#    train_data, valid_data = spensum.dataio.read_train_and_validation_data(
#        args.train, args.valid, file_reader, args.batch_size, gpu=args.gpu)
#
#    pretrained_module = spensum.trainer.pretrain_module(
#        module, train_data, valid_data, args.lr, args.epochs, args.save_module)
#
#    if args.save_predictor is not None:
#        pred_dir = os.path.dirname(args.save_predictor)
#        if pred_dir != "" and not os.path.exists(pred_dir):
#            os.makedirs(pred_dir)
#
#        data = {"model": pretrained_module, "file_reader": file_reader}
#        print("Saving module and file reader...")
#        torch.save(data, args.save_predictor)

if __name__ == "__main__":
    main()
