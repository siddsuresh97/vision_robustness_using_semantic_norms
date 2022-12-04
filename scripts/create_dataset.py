from src.dataset_creation import *

def get_ecoset_concepts():
    # load pdf file from data folder and extract tables
    tables = load_tables_from_pdf('../data/r5_ecoset_Supplemental_Appendix_Table.pdf', '1-end')
    df_ecoset = transform_tables_to_df(tables)
    ecoset_concepts = [i if i not in ['', 'category name', ' '] else i for i in df_ecoset['category name'].values]
    return ecoset_concepts, df_ecoset

def get_leuven_concepts():
    leuven_animals = read_leuven_norms_as_df('../data/LeuvenNorms/ANIMALSexemplarfeaturesbig.txt')
    leuven_artifacts = read_leuven_norms_as_df('../data/LeuvenNorms/ARTIFACTSexemplarfeaturesbig.txt')
    leuven_animals_names = list(leuven_animals.columns[2:])
    leuven_artifacts_names = list(leuven_artifacts.columns[2:])
    leuven_concepts = leuven_animals_names + leuven_artifacts_names
    return leuven_concepts

def get_overlap_concepts():
    ecoset_concepts, _ = get_ecoset_concepts()
    leuven_concepts = get_leuven_concepts()
    overlap_concepts = [concept for concept in ecoset_concepts if concept in leuven_concepts]
    return overlap_concepts

_, df_ecoset = get_ecoset_concepts()

overlap_concepts = get_overlap_concepts()


make_dataset = True
dataset_dir = '/media/external/siddsuresh97/datasets/ecoset'
# new_dataset_dir = '/media/external/siddsuresh97/datasets/ecoset_leuven'
# create_dataset(dataset_dir, new_dataset_dir, overlap_concepts, make_dataset = True)

new_dataset_dir = '/media/external/siddsuresh97/datasets/ecoset_leuven_updated'
create_dataset_updated(dataset_dir, new_dataset_dir, overlap_concepts, make_dataset = True)