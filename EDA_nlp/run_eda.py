from eda import *
from tqdm import tqdm
import pandas as pd
import csv
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--input",
                required=True,
                type=str,
                help="input file of unaugmented data")
ap.add_argument("--output",
                required=False,
                type=str,
                help="output file of unaugmented data")
ap.add_argument("--num_aug",
                required=False,
                default=8,
                type=int,
                help="number of augmented sentences per original sentence")
ap.add_argument("--alpha",
                required=False,
                default=0.1,
                type=float,
                help="percent of words in each sentence to be changed")
args = ap.parse_args()

#the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))


def gen_eda_pair(train_orig, output_file, alpha, num_aug=8):
    """根据输入文件格式调整内容。句子对。"""
    lines = pd.read_csv(train_orig)
    out = open(output_file,"a", newline="")

    csv_write = csv.writer(out)
    csv_write.writerow(['sentence_a','sentence_b','category'])

    eda = EDAnlp()
    for i, line in tqdm(lines.iterrows()):
        try:
            sentence_a = str(line['sentence_a'])
            sentence_b = str(line['sentence_b'])
            category = str(line['category'])
            aug_sentence_a = eda.eda(sentence_a,
                                    alpha_sr=alpha,
                                    alpha_ri=alpha,
                                    alpha_rs=alpha,
                                    p_rd=alpha,
                                    num_aug=num_aug)
            aug_sentencd_b = eda.eda(sentence_b,
                                    alpha_sr=alpha,
                                    alpha_ri=alpha,
                                    alpha_rs=alpha,
                                    p_rd=alpha,
                                    num_aug=num_aug)
            for aug_sentence_a, aug_sentence_b in zip(aug_sentence_a, aug_sentencd_b):
                csv_write.writerow([aug_sentence_a,aug_sentence_b,category])
        except IndexError:
            print("Index Error for sample " + str(i))

    print("generated augmented sentences with eda for " + train_orig + " to " +
          output_file + " with num_aug=" + str(num_aug))



def gen_eda_single(train_orig, output_file, alpha, num_aug=9):
    "根据输入文件格式调整内容。单句。"
    lines = pd.read_csv(train_orig)
    out = open(output_file,"a", newline="")

    csv_write = csv.writer(out)
    csv_write.writerow(['sentence','category'])

    eda = EDAnlp()
    for i, line in tqdm(lines.iterrows()):
        try:
            sentence = str(line['sentence'])
            category = str(line['category'])
            aug_sentence = eda.eda(sentence,
                                    alpha_sr=alpha,
                                    alpha_ri=alpha,
                                    alpha_rs=alpha,
                                    p_rd=alpha,
                                    num_aug=num_aug)
            for aug_sentence in aug_sentence:
                csv_write.writerow([aug_sentence, category])
        except IndexError:
            print("Index Error for sample " + str(i))

    print("generated augmented sentences with eda for " + train_orig + " to " +
          output_file + " with num_aug=" + str(num_aug))


if __name__ == "__main__":
    # gen_eda_pair(args.input, output, alpha=args.alpha, num_aug=args.num_aug)
    gen_eda_single(args.input, output, alpha=args.alpha, num_aug=args.num_aug)