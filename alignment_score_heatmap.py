import click
import itertools
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from pathlib import Path
from typing import Optional
from collections import defaultdict
from Bio import SeqIO, pairwise2, Seq


@click.command(help="Conducts global pairwise alignments against all records in an input FASTAS file, then generates "
                    "a clustermap of pairwise alignment scores. Optionally takes a tab-delimited coding file as input "
                    "to provide additional annotation on the clustermap. Note that this program expects nucleotide "
                    "sequences.")
@click.option('-f', '--fasta', required=True, help='Single input .FASTA file containing nucleotide records to compare.')
@click.option('-o', '--outdir', required=True, help='Path to desired output directory.')
@click.option('-c', '--coding-file', required=False, default=None,
              help='Tab-delimited, two column file containing sequence record names and associated metadata, '
                   'e.g. expected genotype. This file will add additional values to the final clustermap. '
                   'A header row is expected, but the names for the two columns do not matter.')
def cli(fasta, outdir, coding_file):
    fasta = Path(fasta)
    outdir = Path(outdir)
    if coding_file is not None:
        coding_file = Path(coding_file)
    compare(fasta, outdir, coding_file)


def fasta_to_dict(fasta: Path) -> dict:
    f = open(str(fasta))
    fasta_dict = SeqIO.to_dict(SeqIO.parse(f, "fasta"))
    fasta_dict = dict(sorted(fasta_dict.items()))
    return fasta_dict


def generate_sample_pairs(fasta_dict: dict) -> [tuple]:
    sample_pairs = list(itertools.combinations_with_replacement(fasta_dict.keys(), 2))
    return sample_pairs


def get_sample_pair_sequences(fasta_dict: dict, pair: tuple) -> tuple:
    seq1 = fasta_dict[pair[0]]
    seq2 = fasta_dict[pair[1]]
    return seq1, seq2


def translate_sequence(sample: SeqIO.SeqRecord, translation_code: int = 11) -> tuple:
    return sample.seq.translate(translation_code)


def get_alignment_score(seq1: Seq, seq2: Seq) -> float:
    alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
    alignment_score = alignments[0][2]
    return alignment_score


def do_pairwise_alignments(sample_pairs: [tuple], fasta_dict: dict) -> dict:
    alignment_score_dict = defaultdict(dict)
    for pair in tqdm(sample_pairs):
        seq1, seq2 = get_sample_pair_sequences(fasta_dict, pair)
        seq1 = translate_sequence(seq1)
        seq2 = translate_sequence(seq2)
        alignment_score = get_alignment_score(seq1, seq2)
        alignment_score_dict[pair[0]][pair[1]] = alignment_score
        alignment_score_dict[pair[1]][pair[0]] = alignment_score
    return alignment_score_dict


def parse_coding_file(coding_file: Path) -> dict:
    """
    Expects a tab delimited, two column text file. Column names are arbitrary and don't matter.
    Example:

    sample  coding_val
    1       blue
    2       blue
    3       red
    4       blue

    """
    df = pd.read_csv(coding_file, delimiter="\t", index_col=0)
    coding_dict = df.T.to_dict(orient='records')[0]
    return coding_dict


def generate_color_coding_dict(coding_dict: dict):
    coding_values = list(set([i for i in coding_dict.values()]))
    colors = sns.color_palette("Paired", len(coding_values))
    color_dict = dict(zip(coding_values, colors))
    return color_dict


def compare(fasta: Path, outdir: Path, coding_file: Optional[Path]):
    if coding_file:
        row_colors = True
    else:
        row_colors = False

    fasta_dict = fasta_to_dict(fasta)
    print(f"Analyzing {len(fasta_dict.keys())} total samples")

    # Conduct pairwise global alignments
    sample_pairs = generate_sample_pairs(fasta_dict)
    alignment_score_dict = do_pairwise_alignments(sample_pairs, fasta_dict)

    df = pd.DataFrame(alignment_score_dict)
    out_csv = outdir / fasta.with_suffix(".csv").name
    print(f"Writing csv to {out_csv}")
    df.to_csv(out_csv)

    # Generate clustermap image
    out_png = outdir / fasta.with_suffix(".png").name
    print(f"Writing clustermap to {out_png}")
    if row_colors:
        coding_dict = parse_coding_file(coding_file)
        color_dict = generate_color_coding_dict(coding_dict)
        colors_series = [color_dict[coding_dict[x]] for x in list(df.index.values)]
        row_colors = pd.Series(colors_series, index=df.index.values)
        sns_plot = sns.clustermap(df, method="single", figsize=(32, 32), col_colors=row_colors)
        for label in color_dict.keys():
            sns_plot.ax_col_dendrogram.bar(0, 0, color=color_dict[label], label=label, linewidth=0)
        sns_plot.ax_col_dendrogram.legend(loc="center", ncol=5)
        sns_plot.savefig(out_png)
    else:
        sns_plot = sns.clustermap(df, method="single", figsize=(32, 32))
        sns_plot.savefig(out_png)


if __name__ == "__main__":
    cli()