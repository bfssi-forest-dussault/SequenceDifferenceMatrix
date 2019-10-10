import click
import itertools
import pandas as pd
import seaborn as sns

from tqdm import tqdm
from pathlib import Path
from typing import Optional
from collections import defaultdict
from Bio import SeqIO, pairwise2, Seq


@click.command(help="Conducts global pairwise alignments against all records in an input FASTA file, then generates "
                    "a clustermap of pairwise alignment scores. Optionally takes a tab-delimited coding file as input "
                    "to provide additional annotation on the clustermap. Note that this program expects nucleotide "
                    "sequences.")
@click.option('-f', '--fasta',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help='Single input .FASTA file containing nucleotide records to compare.')
@click.option('-o', '--outdir',
              type=click.Path(),
              required=True,
              help='Path to desired output directory.')
@click.option('-c', '--coding-file',
              required=False,
              type=click.Path(exists=True, dir_okay=False),
              default=None,
              help='Tab-delimited, two column file containing sequence record names and associated metadata, '
                   'e.g. expected genotype. This file will add additional values to the final clustermap. '
                   'A header row is expected, but the names for the two columns do not matter.')
def cli(fasta, outdir, coding_file):
    """ CLI interface """
    fasta = Path(fasta)
    outdir = Path(outdir)
    if coding_file is not None:
        coding_file = Path(coding_file)
    alignment_score_clustermap(fasta, outdir, coding_file)


def fasta_to_dict(fasta: Path) -> dict:
    """ Parses fasta file with BioPython, return dict containing {fasta_header:SeqRecord} structure """
    f = open(str(fasta))
    fasta_dict = SeqIO.to_dict(SeqIO.parse(f, "fasta"))
    fasta_dict = dict(sorted(fasta_dict.items()))
    return fasta_dict


def generate_sample_pairs(fasta_dict: dict) -> [tuple]:
    """ Generates pairs (with replacement) for every key (i.e. FASTA record header) in fasta_dict """
    sample_pairs = list(itertools.combinations_with_replacement(fasta_dict.keys(), 2))
    return sample_pairs


def get_sample_pair_sequences(fasta_dict: dict, pair: tuple) -> tuple:
    """ Retrieve the sequence records from fasta_dict for a given pair of records """
    seq1 = fasta_dict[pair[0]]
    seq2 = fasta_dict[pair[1]]
    return seq1, seq2


def translate_sequence(sample: SeqIO.SeqRecord, translation_code: int = 11) -> tuple:
    """ Translates nt sequence according to translation code, which defaults to Bacterial (11) """
    return sample.seq.translate(translation_code)


def get_alignment_score(seq1: Seq, seq2: Seq) -> float:
    """ Conducts globalxx alignment between two sequences with Biopython and returns the alignment score"""
    alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
    alignment_score = alignments[0][2]
    return alignment_score


def do_pairwise_alignments(sample_pairs: [tuple], fasta_dict: dict) -> dict:
    """ Iterates over each sample record pair and generates a defaultdict containing every pair aligment scores """
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
    """ Creates a dict associating every unique value in the coding dict with a unique colour """
    coding_values = list(set([i for i in coding_dict.values()]))
    colors = sns.color_palette("Paired", len(coding_values))
    color_dict = dict(zip(coding_values, colors))
    return color_dict


def alignment_score_clustermap(fasta: Path, outdir: Path, coding_file: Optional[Path]):
    """ Main method call for conducting pairwise comparisons of all records within FASTA, generating clustermap """

    fasta_dict = fasta_to_dict(fasta)
    print(f"Analyzing {len(fasta_dict.keys())} total samples")

    # Conduct pairwise global alignments
    sample_pairs = generate_sample_pairs(fasta_dict)
    alignment_score_dict = do_pairwise_alignments(sample_pairs, fasta_dict)

    df = pd.DataFrame(alignment_score_dict)
    out_csv = outdir / fasta.with_suffix(".csv").name
    print(f"Writing csv to {out_csv}")
    df.to_csv(out_csv)

    out_png = outdir / fasta.with_suffix(".png").name
    print(f"Writing clustermap to {out_png}")
    # If the coding file was provided, attempt to add metadata to clustermap
    if coding_file:
        coding_dict = parse_coding_file(coding_file)
        color_dict = generate_color_coding_dict(coding_dict)
        colors_series = [color_dict[coding_dict[x]] for x in list(df.index.values)]
        row_colors = pd.Series(colors_series, index=df.index.values)
        sns_plot = sns.clustermap(df, method="single", figsize=(32, 32), col_colors=row_colors)
        for label in color_dict.keys():
            sns_plot.ax_col_dendrogram.bar(0, 0, color=color_dict[label], label=label, linewidth=0)
        sns_plot.ax_col_dendrogram.legend(loc="center", ncol=5)
        sns_plot.savefig(out_png)
    # Generate simple clustermap
    else:
        sns_plot = sns.clustermap(df, method="single", figsize=(32, 32))
        sns_plot.savefig(out_png)


if __name__ == "__main__":
    cli()
