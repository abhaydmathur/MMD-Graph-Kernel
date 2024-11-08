from kernels import MMDGK, deep_MMDGK
from utils import arg_parse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# datasets = ["MUTAG", "BZR", "DHFR", "PROTEINS", "PTC_FM"] #, 'Fingerprint', 'BZR_MD']
datasets = ["Fingerprint", "BZR_MD"]

models = ["vanilla", "deep"]
# models = ['deep']

onas = [True, False]

# dis_gammas = [1e-1, 1e0, 1e1]

# alphas = [1e-1, 1e0, 1e1]

# objective_functions = ["KL", "SCL", "UCL"]

if __name__ == "__main__":

    args = arg_parse()

    for m in models:
        args.model = m
        for d in datasets:
            args.dataname = d
            for ona in onas:
                args.only_node_attr = ona
                # for dis_gamma in dis_gammas:
                #     args.dis_gamma = dis_gamma
                # for alpha in alphas:
                #     args.alpha = alpha
                # for objective_function in objective_functions:
                #     args.objective_fuc = objective_function
                print(
                    f"Model: {args.model}, Dataset: {args.dataname}, Only Node Attr: {args.only_node_attr}"
                )

                try:
                    if args.model == "vanilla":
                        kernel = MMDGK(args)
                    else:
                        kernel = deep_MMDGK(args)
                except Exception as e:
                    print(e)
                    continue
