from bleu import *
import argparse
import os
import codecs

parser = argparse.ArgumentParser(
    description="Calculate BLEU scores for the input hypothesis and reference files")
parser.add_argument(
    "-hyp",
    nargs=1,
    dest="pf_hypothesis",
    type=str,
    help="The path of the hypothesis file.")
parser.add_argument(
    "-ref",
    nargs='+',
    dest="pf_references",
    type=str,
    help="The path of the references files.")
args = parser.parse_args()
def file_exist(pf):
    if os.path.isfile(pf):
        return True
    return False

if args.pf_hypothesis!=None or args.pf_references!=None:
    if args.pf_hypothesis==None:
        raise Exception("Missing references files...")
    if args.pf_references==None:
        raise Exception("Missing hypothesis files...")

    n = None
    data = []
    for pf in args.pf_hypothesis+args.pf_references:
        if not file_exist(pf):
            raise Exception("File Not Found: %s"%(pf))
        f = codecs.open(pf, encoding="utf-8")
        data.append(f.readlines())
        if n==None:
            n = len(data[-1])
        elif n != len(data[-1]):
            raise Exception("Not parrallel: %s %d-%d"%(pf, n, len(data[-1])))
        f.close()

    hyp_data = data[0]
    ref_data = list(map(list, zip(*data[1:])))

    bleu, addition = corpus_bleu(hyp_data, ref_data)

    print("BLEU = %.2f, %.1f/%.1f/%.1f/%.1f (BP=%.3f, ratio=%.3f, hyp_len=%d, ref_len=%d)"%(bleu[0]*100, bleu[1]*100, bleu[2]*100, bleu[3]*100, bleu[4]*100, addition[0], addition[1], addition[2], addition[3]))
