
import pandas as pd
import numpy as np

from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from rdkit.Chem import Descriptors

from rdkit import Chem
from rdkit.Chem import AllChem


def ecfp_featurizer(train, test):
    """Creates datasets ready to input into ML models
       Uses mol2vec to create ECFP features from SMILES strings"""
    
    # convert SMILES to RDKit Mol object
    train['mol'] = train['std_compounds'].apply(lambda x: Chem.MolFromSmiles(x))
    test['mol'] = test['std_compounds'].apply(lambda x: Chem.MolFromSmiles(x))
    
    model = word2vec.Word2Vec.load('model_300dim.pkl')
    
    #Constructing sentences
    train['sentence'] = train.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)
    test['sentence'] = test.apply(lambda x: MolSentence(mol2alt_sentence(x['mol'], 1)), axis=1)

    # extracting embeddings to a numpy.array
    # note that we always should mark unseen='UNK' in sentence2vec() 
    # so that model is taught how to handle unknown substructures
    train['mol2vec'] = [DfVec(x) for x in sentences2vec(train['sentence'], model, unseen='UNK')]
    test['mol2vec'] = [DfVec(x) for x in sentences2vec(test['sentence'], model, unseen='UNK')]
    x_train = np.array([x.vec for x in train['mol2vec']])
    x_test = np.array([x.vec for x in test['mol2vec']])
    
    return x_train, x_test


def number_of_atoms(atom_list, df):
    """Helper function for oned_featurizer"""
    
    for i in atom_list:
        df['num_of_{}_atoms'.format(i)] = df['mol'].apply(lambda x: len(x.GetSubstructMatches(Chem.MolFromSmiles(i))))
        
def oned_featurizer(train, test):
    """Creates datasets ready to input into ML models
       Uses mol2vec to create 1D representations of molecules from SMILES strings
       Includes the following features: number of atoms, number of heavy atoms,
           number of C, O, N, and Cl atoms, molecular weight, 
           number of valence electrons, and number of heteroatoms"""
    
    # convert SMILES to RDKit Mol object
    train['mol'] = train['std_compounds'].apply(lambda x: Chem.MolFromSmiles(x))
    test['mol'] = test['std_compounds'].apply(lambda x: Chem.MolFromSmiles(x))
    
    # number of atoms
    train['mol'] = train['mol'].apply(lambda x: Chem.AddHs(x))
    train['num_of_atoms'] = train['mol'].apply(lambda x: x.GetNumAtoms())
    train['num_of_heavy_atoms'] = train['mol'].apply(lambda x: x.GetNumHeavyAtoms())
    number_of_atoms(['C','O', 'N', 'Cl'], train)
    
    test['mol'] = test['mol'].apply(lambda x: Chem.AddHs(x))
    test['num_of_atoms'] = test['mol'].apply(lambda x: x.GetNumAtoms())
    test['num_of_heavy_atoms'] = train['mol'].apply(lambda x: x.GetNumHeavyAtoms())
    number_of_atoms(['C','O', 'N', 'Cl'], test)
    
    # molecular descriptors
    train['mol_w'] = train['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
    train['num_valence_electrons'] = train['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
    train['num_heteroatoms'] = train['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
    
    test['mol_w'] = test['mol'].apply(lambda x: Descriptors.ExactMolWt(x))
    test['num_valence_electrons'] = test['mol'].apply(lambda x: Descriptors.NumValenceElectrons(x))
    test['num_heteroatoms'] = test['mol'].apply(lambda x: Descriptors.NumHeteroatoms(x))
    
    x_train = train[['num_of_atoms', 'num_of_heavy_atoms', 'num_of_C_atoms', 
                     'num_of_O_atoms', 'num_of_N_atoms', 'num_of_Cl_atoms', 'mol_w', 
                     'num_valence_electrons', 'num_heteroatoms']]
    x_test = test[['num_of_atoms', 'num_of_heavy_atoms', 'num_of_C_atoms', 
                     'num_of_O_atoms', 'num_of_N_atoms', 'num_of_Cl_atoms', 'mol_w', 
                     'num_valence_electrons', 'num_heteroatoms']]
    
    return x_train, x_test
