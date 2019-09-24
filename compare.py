import itertools
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO, pairwise2
from collections import defaultdict
import seaborn as sns


def compare(input_fasta: Path, outdir: Path):
    f = open(str(input_fasta))
    fasta_dict = SeqIO.to_dict(SeqIO.parse(f, "fasta"))

    # Conduct pairwise global alignments
    d = defaultdict(dict)
    sample_pairs = list(itertools.combinations_with_replacement(fasta_dict.keys(), 2))
    for pair in tqdm(sample_pairs):  # Taking the first 10 for debugging
        o = {}
        seq1 = fasta_dict[pair[0]].seq.translate(11)
        seq2 = fasta_dict[pair[1]].seq.translate(11)
        alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
        sample_1 = pair[0]
        sample_2 = pair[1]
        alignment_score = alignments[0][2]
        d[sample_1][sample_2] = alignment_score
        d[sample_2][sample_1] = alignment_score

    # .csv
    df = pd.DataFrame(d)
    out_csv = outdir / input_fasta.with_suffix(".csv").name
    print(f"Writing csv to {out_csv}")
    df.to_csv(out_csv)

    # Excel
    out_xlsx = outdir / input_fasta.with_suffix(".xlsx").name
    print(f"Writing Excel to {out_xlsx}")
    writer = pd.ExcelWriter(str(out_xlsx), engine='xlsxwriter')
    df.to_excel(writer, sheet_name="SequenceDifferenceMatrix")
    worksheet = writer.sheets['SequenceDifferenceMatrix']
    worksheet.conditional_format('B2:AMJ100000', {'type': '3_color_scale'})
    writer.save()

    # Generate heatmap image
    out_png = outdir / input_fasta.with_suffix(".png").name
    print(f"Writing heatmap to {out_png}")
    sns_plot = sns.heatmap(df, annot=False)
    fig = sns_plot.get_figure()
    fig.tight_layout()
    fig.savefig(out_png)


if __name__ == "__main__":
    aligned_fasta = Path(
        "/home/forest/Documents/AlexGill_Projects/Sept2019StxTrees/matrices/Stx2A_all_samples.aligned.translated.fasta")
    fasta = Path("/home/forest/Documents/AlexGill_Projects/Sept2019StxTrees/stxA_tree/Stx2A_all_samples.fasta")
    outdir = Path("/home/forest/Documents/AlexGill_Projects/Sept2019StxTrees/")
    compare(input_fasta=fasta, outdir=outdir)
