import argparse
import logging
import os
from Bio import SeqIO
from shutil import copyfile
import subprocess as sp
import collections
from random import shuffle

nucleotides = ['A', 'T', 'C', 'G']


# ---------------------------------------------------------

def init():
    parser = argparse.ArgumentParser(prog='what.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sample_id', help='subfolder id')
    # parser.add_argument('-sample_id_map', help='map file to get sample ids to use')
    parser.add_argument('-output_dir', help='path to output folder')
    parser.add_argument('-genome_file', help='path to hg19 genome fasta')
    parser.add_argument('-hg_bed', help='path to hg bed file')
    parser.add_argument('-motif_list', help='path to motif list file')
    parser.add_argument('-subsampling_rate', help='0.2, 0.15, 0.1')
    parser.add_argument('-samplenum', help='0, 1, 2, 3')
    args = parser.parse_args()

    if not args.sample_id or not args.output_dir or not args.genome_file or not args.hg_bed or not args.motif_list or not args.subsampling_rate or not args.samplenum:
        parser.print_help()
        return 0

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)

    return args


# ---------------------------------------------------------

def countSubtitutionType(found_nv, chr, ref, alt, pos, motifs_dict):
    base_nuc = str(ref[len(ref) - 1])
    alt_nuc = str(alt)
    final_ref = ""
    final_alt = ""
    final_nv = ""
    actual_motif = found_nv + ".." + str(found_nv[0] + alt_nuc + found_nv[2])
    motifs = list(motifs_dict.keys())
    if actual_motif not in motifs:
        final_nv_elements = []
        for fnv_ in found_nv:
            if fnv_ == "G":
                final_nv_elements.append("C")
            if fnv_ == "C":
                final_nv_elements.append("G")
            if fnv_ == "A":
                final_nv_elements.append("T")
            if fnv_ == "T":
                final_nv_elements.append("A")
        final_nv = ''.join(final_nv_elements)

        final_alt_elements = []
        for a_ in alt_nuc:
            if a_ == "G":
                final_alt_elements.append("C")
            if a_ == "C":
                final_alt_elements.append("G")
            if a_ == "A":
                final_alt_elements.append("T")
            if a_ == "T":
                final_alt_elements.append("A")
        final_alt = ''.join(final_alt_elements)

        final_ref_elements = []
        for r_ in str(ref):
            if r_ == "G":
                final_ref_elements.append("C")
            if r_ == "C":
                final_ref_elements.append("G")
            if r_ == "A":
                final_ref_elements.append("T")
            if r_ == "T":
                final_ref_elements.append("A")
        final_ref = ''.join(final_ref_elements)
    else:
        final_nv = found_nv
        final_alt = str(alt_nuc)
        final_ref = str(ref)

    key_ = str(final_nv[0] + final_ref + final_nv[len(final_nv) - 1] + ".." + final_nv[0] + final_alt + final_nv[
        len(final_nv) - 1])
    if key_ not in motifs_dict:
        print("Key is not found among motifs")
    else:
        motifs_dict[key_] = motifs_dict[key_] + 1

    return motifs_dict


# ---------------------------------------------------------


def getBinCount(bin_fields, bins_with_counts, pos, chr):
    for chr_ in bin_fields:
        if chr_ != chr:
            continue
        names_with_fields = bin_fields[chr_]
        for bin_ in names_with_fields:
            region = names_with_fields[bin_]
            if region[1] > pos and pos >= region[0]:
                bins_with_counts[bin_] = bins_with_counts[bin_] + 1
    return bins_with_counts


# ---------------------------------------------------------

def remove_percentage(list_a, percentage):
    shuffle(list_a)
    count = int(len(list_a) * percentage)
    if not count: return []  # edge case, no elements removed
    list_a[-count:], list_b = [], list_a[-count:]
    return list_b


# ---------------------------------------------------------

def obtainBinData(driver_gene_fields, fasta_records, sid, motifs, bin_fields, vcf, output_dir,
                  subsampling_rate, samplenum):
    # variant_classification = ["Intron", "5'UTR", "3'UTR", "5'Flank", "3'Flank", "IGR"]
    pcawg_sample_id = str(sid + "_" + str(samplenum))
    outfile = os.path.join(output_dir, "{}_mut_freq_bins.txt".format(pcawg_sample_id))
    outfile_96motif_dbsnp = os.path.join(output_dir,
                                         "{}_motifs_96.txt".format(pcawg_sample_id))
    outfile_driver_dbsnp = os.path.join(output_dir,
                                        "{}_driver.txt".format(pcawg_sample_id))
    # outfile_96motif = os.path.join(output_dir, "{}_mut_trinuc_96.txt".format(pcawg_sample_id))
    # outfile_96motif_dbsnp = os.path.join(output_dir, "{}_mut_trinuc_96_dbsnp.txt".format(pcawg_sample_id))

    print(list(motifs.keys()))
    bins_with_counts = dict()
    bins_with_counts_dbsnp = dict()
    for chr_ in bin_fields:
        bin_fields_chr = bin_fields[chr_]
        for b_f_ in bin_fields_chr:
            bins_with_counts[b_f_] = 0
            bins_with_counts_dbsnp[b_f_] = 0

    drivers = dict()
    for df in driver_gene_fields:
        drivers[df] = 0

    vcf_file = open(vcf, "r")
    vcf_file_ = vcf_file.readlines()
    vcfdata = [line_ for line_ in vcf_file_ if not line_.startswith("#")]
    filtered_vcfdata = remove_percentage(vcfdata, subsampling_rate)

    for vx, variant_line in enumerate(filtered_vcfdata):
        # if variant_line.startswith("#"):
        #     continue
        variant = variant_line.strip("\n").split("\t")
        chr = str("chr" + variant[0])
        if "M" in chr or "X" in chr or "Y" in chr:
            continue
        pos = int(variant[1]) - 1
        ref = variant[3]
        alt = variant[4]
        info = variant[7]
        if len(alt) == len(ref) and len(ref) == 1:
            # and filter == "PASS"
            bins_with_counts = getBinCount(bin_fields, bins_with_counts, pos, chr)
            left, right = getFlankingBases(chr, pos, fasta_records)
            found_snv = ''.join([left, ref, right])
            motifs = countSubtitutionType(found_snv, chr, ref, alt, pos, motifs)

            ### check if SNV is driver
            for dgf in driver_gene_fields:
                splitdgfname = dgf.split("_")
                chrom_ = str("chr" + splitdgfname[len(splitdgfname) - 1])
                if chrom_ == chr:
                    region_borders = driver_gene_fields[dgf]
                    if region_borders[1] >= pos and pos >= region_borders[0]:
                        drivers[dgf] = drivers[dgf] + 1

    out_f = open(outfile, "w")
    out_f.write("sampleId,bin,n" + "\n")
    bins_with_counts_od = collections.OrderedDict(sorted(bins_with_counts.items()))
    for bin_name in bins_with_counts_od:
        bin_count = bins_with_counts_od[bin_name]
        out_f.write(str(pcawg_sample_id) + "," + str(bin_name) + "," + str(bin_count))
        out_f.write("\n")
    out_f.close()

    out_f_96motif = open(outfile_96motif_dbsnp, "w")
    out_f_96motif.write("sampleId,motif,n" + "\n")
    motifs_sorted = collections.OrderedDict(sorted(motifs.items()))
    for m_ in motifs_sorted:
        out_f_96motif.write(str(pcawg_sample_id) + "," + str(m_) + "," + str(motifs_sorted[m_]))
        out_f_96motif.write("\n")
    out_f_96motif.close()

    out_f_driver_dbsnp = open(outfile_driver_dbsnp, "w")
    out_f_driver_dbsnp.write("sampleId,driver,n" + "\n")
    drivers_sorted = collections.OrderedDict(sorted(drivers.items()))
    for d_ in drivers_sorted:
        out_f_driver_dbsnp.write(str(pcawg_sample_id) + "," + str(d_) + "," + str(drivers_sorted[d_]))
        out_f_driver_dbsnp.write("\n")
    out_f_driver_dbsnp.close()


# ---------------------------------------------------------

def parseMotifList(motif_file):
    motif_data = open(motif_file, "r")
    motif_list = dict()
    line_counter = 0
    for line in motif_data:
        if line_counter == 0:
            line_counter += 1
        else:
            lsplit = line.strip("\n").split("..")
            ref = lsplit[0]
            alt = lsplit[1]
            motif = ref + ".." + alt
            motif_list[motif] = 0
    return motif_list


# ---------------------------------------------------------

def obtain96ProfilePerSample(driver_gene_fields, fasta_records, bin_fields, motif_list, sample_folder, output_dir,
                             subsampling_rate, samplenum):
    all_files = os.listdir(sample_folder)
    if int(samplenum) == 0:
        source_vcf = ''.join([os.path.join(sample_folder, i) for i in all_files if i.endswith(".somatic.snv_mnv_PASS.vcf.gz")])
        copyfile(source_vcf,[os.path.join(args.output_dir, i) for i in all_files if i.endswith(".somatic.snv_mnv_PASS.vcf.gz")][0])
        vcf = [os.path.join(args.output_dir, i) for i in all_files if i.endswith(".somatic.snv_mnv_PASS.vcf.gz")][0]
        vcf = unzipGz(vcf)
    else:
        source_vcf = ''.join(
            [os.path.join(sample_folder, i) for i in all_files if i.endswith(".somatic.snv_mnv_PASS.vcf.gz")])
        copyfile(source_vcf,
                 [os.path.join(args.output_dir, i) for i in all_files if i.endswith(".somatic.snv_mnv_PASS.vcf.gz")][0])
        vcf = [os.path.join(args.output_dir, i) for i in all_files if i.endswith(".somatic.snv_mnv_PASS.vcf.gz")][0]
        vcf = vcf[:-3]
    motifs = parseMotifList(motif_list)
    if vcf.endswith("gz"):
        vcf = unzipGz(vcf)
    head, tail = os.path.split(vcf)
    split_tail = tail.split(".")
    pcawg_sample_id = split_tail[0]
    obtainBinData(driver_gene_fields, fasta_records, pcawg_sample_id, motifs, bin_fields, vcf, output_dir,
                  subsampling_rate, samplenum)
    print(str("Extraction from " + pcawg_sample_id + " is done"))


# ---------------------------------------------------------

def unzipGz(filename):
    if filename.endswith('.gz'):
        sp.check_call(['gzip', '-d', filename])
        return filename[:-3]
    else:
        return filename


# ---------------------------------------------------------

def getFlankingBases(chr, pos, fasta_records):
    for seq_id in fasta_records:
        seq_record = fasta_records[seq_id]
        if str(seq_id) == chr:
            try:
                left = str(seq_record.seq)[pos - 1]
                right = str(seq_record.seq)[pos + 1]
                return left, right
            except ValueError:
                print("Not valid coordinate")


# ---------------------------------------------------------

def readInBins(hg_bed):
    bin_fields = dict()
    bin_file = open(hg_bed, "r")
    counter = 0
    for bin_ in bin_file:
        if counter == 0:
            counter += 1
        else:
            bin_splitted = bin_.strip("\n").split(" ")
            start = int(bin_splitted[1])
            end = int(bin_splitted[2])
            name = bin_splitted[3]
            chr_ = bin_splitted[0]
            if chr_ not in bin_fields:
                bin_fields[chr_] = dict()
            names_with_fields = bin_fields[chr_]
            names_with_fields[name] = [start, end]
    return bin_fields


# ---------------------------------------------------------

def parseSampleIdMap(sample_id_map):
    mapping = open(sample_id_map, "r")
    ids_to_include = list()
    for m_ in mapping:
        m_split = m_.strip("\n").split(",")
        if m_split[0] not in ids_to_include:
            ids_to_include.append(m_split[0])
    return ids_to_include


# ---------------------------------------------------------

def parseDriverGeneRegions():
    gene_region_file = open("./pcawg_gene_regions.txt", 'r')
    gene_regions = dict()
    for gregline in gene_region_file:
        fields = gregline.strip("\n").split("\t")
        gene = fields[0] + "_" + fields[1]
        region = [int(fields[2]), int(fields[3])]
        gene_regions[gene] = region
    return gene_regions


# ---------------------------------------------------------

def run(args):
    # ids_to_include = parseSampleIdMap(args.sample_id_map)
    fasta_records = dict()
    for record in SeqIO.parse(args.genome_file, "fasta"):
        fasta_records[record.id] = record
        print(record.id)
    bin_fields = readInBins(args.hg_bed)
    driver_gene_fields = parseDriverGeneRegions()
    obtain96ProfilePerSample(driver_gene_fields, fasta_records, bin_fields, args.motif_list, args.sample_id,
                             args.output_dir, float(args.subsampling_rate), args.samplenum)


# ---------------------------------------------------------

args = init()
if args is not 0:
    run(args)