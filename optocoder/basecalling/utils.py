import numpy as np

def map_back(barcode):
    nucs = list(barcode)
    mapper = ['T', 'G', 'C', 'A']
    new_barcode = []
    for nuc in nucs:
        new_barcode.append(str(mapper.index(nuc)))

    return ''.join(new_barcode)

def convert_to_color_space(nuc, next_nuc):
    if nuc == 'A':
        return str(['A', 'C', 'G', 'T', 'N'].index(next_nuc))
    elif nuc == 'C':
        return str(['C', 'A', 'T', 'G', 'N'].index(next_nuc))
    elif nuc == 'G':
        return str(['G', 'T', 'A', 'C', 'N'].index(next_nuc))
    elif nuc == 'T':
        return str(['T', 'G', 'C', 'A', 'N'].index(next_nuc))
    elif nuc == 'N':
        return str(5)

def convert_barcodes(barcode, ligation_seq):
    nucleotides = list(barcode)
    part_1 = nucleotides[:6]
    bc_1 = ['T']
    bc_1.extend(part_1)
    bc_1.extend('T')

    part_2 = nucleotides[6:]
    bc_2 = ['A']
    bc_2.extend(part_2)
    bc_2.extend('')

    bc_1_cs = []
    bc_2_cs = []
    for i in range(len(bc_1) - 1):
        bc_1_cs.extend(convert_to_color_space(bc_1[i], bc_1[i+1]))
    
    for i in range(len(bc_2) - 1):
        bc_2_cs.extend(convert_to_color_space(bc_2[i], bc_2[i+1]))

    bc_1 = list(np.array(bc_1_cs)[ligation_seq])
    bc_2 = list(np.array(bc_2_cs)[ligation_seq])
    bc_1.extend(bc_2)
    return bc_1