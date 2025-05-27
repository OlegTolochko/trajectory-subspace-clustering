import torch
import pandas as pd
import os
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional

from rich.console import Console
import matplotlib.pyplot as plt
import seaborn as sns

from models.trajectory_embedder import TrajectoryEmbeddingModel
from datasets import Hopkins155, KT3DMoSeg, Hopkins12
from inference import evaluate_model_performance

class ModelEvaluator:
    def __init__(self, models_dir = 'out/models', seed = 42):
        self.models_dir = Path(models_dir)
        self.seed = seed
        self.results = []
        self.console = Console()
        
    def parse_model_name(self, model_name):
        name = model_name.replace('.pt', '')
        
        pattern = r'([a-z0-9]+)_(\d+)_(\d+)_(full|split)_(incfeat|exfeat)_(ortho|noortho)(?:_(alph0))?'
        match = re.match(pattern, name)
        
        if not match:
            raise ValueError(f"Model name '{model_name}' doesn't match expected format: "
                            "{{dataset}}_{{pre_epochs}}_{{full_epochs}}_{{scope}}_{{featdiff}}_{{ortho}}[_alph0]")
        dataset, pre_epochs, full_epochs, scope, featdiff, ortho, alph0 = match.groups()

        return {
            'dataset': dataset,
            'pretraining_epochs': int(pre_epochs),
            'full_epochs': int(full_epochs),
            'trained_on_full': scope == 'full',
            'feature_diff_excluded': featdiff == 'exfeat',
            'orthogonality': ortho,
            'alpha_zero': alph0 == 'alph0',
        }
    
    def get_dataset(self, dataset_name):
        if dataset_name == 'hopk155':
            return Hopkins155()
        elif dataset_name == 'kt':
            return KT3DMoSeg()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def get_evaluation_datasets(self, dataset_name):
        eval_datasets = []
        
        if dataset_name == 'hopk155':
            eval_datasets.append(('Hopkins155', Hopkins155()))
            eval_datasets.append(('Hopkins12', Hopkins12()))
        elif dataset_name == 'kt':
            eval_datasets.append(('KT3DMoSeg', KT3DMoSeg()))
        
        return eval_datasets
    
    def get_train_test_split(self, dataset, trained_on_full):
        if trained_on_full:
            return dataset, None
        
        seq_ids = list(range(len(dataset)))
        train_ids, val_ids = train_test_split(
            seq_ids, test_size=0.2, random_state=self.seed, shuffle=True
        )
        
        train_set = torch.utils.data.Subset(dataset, train_ids)
        test_set = torch.utils.data.Subset(dataset, val_ids)
        
        return train_set, test_set
    
    def load_model(self, model_path):
        model = TrajectoryEmbeddingModel()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        
        return model
    
    def test_orthogonality(self, model):
        subspace_estimator = model.subspace_estimator
        device = next(subspace_estimator.parameters()).device

        seq_length = 300
        t_vector = torch.arange(seq_length, dtype=torch.float32).to(device)
        t_vector_batch = t_vector.unsqueeze(0)

        with torch.no_grad():
            h_t_values = subspace_estimator.calculate_basis_functions(t_vector_batch)

        basis_function_vectors = h_t_values.squeeze(0).cpu().numpy()
        N_basis_functions = basis_function_vectors.shape[1]

        normalized_basis_vectors = basis_function_vectors / (np.linalg.norm(basis_function_vectors, axis=0, keepdims=True) + 1e-9)
        cosine_sim_matrix = normalized_basis_vectors.T @ normalized_basis_vectors

        diag_mask = ~np.eye(N_basis_functions, dtype=bool) 
        mean_abs_off_diagonal_cosine_sim = np.mean(np.abs(cosine_sim_matrix[diag_mask]))
        return mean_abs_off_diagonal_cosine_sim

    def evaluate_single_model(self, model_path, include_orthogonality=True):
        model_name = Path(model_path).name
        print(f"\nEvaluating model: {model_name}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        config = self.parse_model_name(model_name)
        model = self.load_model(model_path)
        train_dataset = self.get_dataset(config['dataset'])
        
        train_set, test_set = self.get_train_test_split(
            train_dataset, config['trained_on_full']
        )
        eval_datasets = self.get_evaluation_datasets(config['dataset'])
        
        results = {
            'model_name': model_name,
            'dataset': config['dataset'],
            'pretraining_epochs': config['pretraining_epochs'],
            'full_epochs': config['full_epochs'],
            'trained_on_full': config['trained_on_full'],
            'feature_diff_excluded': config['feature_diff_excluded'],
            'alpha_zero': config['alpha_zero'],
            'orthogonality': config['orthogonality']
        }
        
        if include_orthogonality:
            try:
                orthogonality_score = self.test_orthogonality(model)
                results['orthogonality_measure'] = orthogonality_score
                print(f"Orthogonality measure: {orthogonality_score:.4f}")
            except Exception as e:
                print(f"Failed to calculate orthogonality measure: {e}")
                results['orthogonality_measure'] = None
    
        algorithms = ['hierarchical', 'kmeans', 'spectral']
        print(f"Evaluating on training data...")
        
        train_loader = DataLoader(train_set, batch_size=1)
        
        for algo in algorithms:
            error_rate = evaluate_model_performance(
                model, train_loader, algo, device
            )
            results[f'train_{algo}_error'] = error_rate
        
        if test_set is not None:
            print(f"Evaluating on test data...")
            test_loader = DataLoader(test_set, batch_size=1)
            
            for algo in algorithms:
                error_rate = evaluate_model_performance(
                    model, test_loader, algo, device
                )
                results[f'test_{algo}_error'] = error_rate
        else:
            for algo in algorithms:
                results[f'test_{algo}_error'] = None
        
        for eval_name, eval_dataset in eval_datasets:
            if eval_name != config['dataset'].upper():
                print(f"Evaluating on {eval_name}...")
                eval_loader = DataLoader(eval_dataset, batch_size=1)
                
                for algo in algorithms:
                    error_rate = evaluate_model_performance(
                        model, eval_loader, algo, device
                    )
                    results[f'{eval_name.lower()}_{algo}_error'] = error_rate
        
        return results
    
    def evaluate_models(self, model_names=None, include_orthogonality=True):
        if model_names is None:
            model_files = list(self.models_dir.glob('*.pt'))
            model_names = [f.name for f in model_files]
        
        print(f"Found {len(model_names)} models to evaluate")
        
        all_results = []
        
        for model_name in model_names:
            model_path = self.models_dir / model_name
            if not model_path.exists():
                print(f"Model file not found: {model_path}")
                continue
                
            result = self.evaluate_single_model(str(model_path), include_orthogonality)
            if result is not None:
                all_results.append(result)
        
        df = pd.DataFrame(all_results)
        algorithms = ['hierarchical', 'kmeans', 'spectral']
        
        for algo in algorithms:
            train_col = f'train_{algo}_error'
            if train_col in df.columns:
                df[f'train_{algo}_mean'] = df[train_col]
                df[f'train_{algo}_median'] = df[train_col]
            
            test_col = f'test_{algo}_error'
            if test_col in df.columns:
                df[f'test_{algo}_mean'] = df[test_col]
                df[f'test_{algo}_median'] = df[test_col]
    
        return df
    
    def create_comparison_table(self, df):
        if df.empty:
            return df
            
        comparison_cols = ['model_name', 'dataset', 'pretraining_epochs', 'full_epochs', 'trained_on_full', 
                          'orthogonality', 'feature_diff_excluded', 'alpha_zero']
        
        if 'orthogonality_measure' in df.columns:
            comparison_cols.append('orthogonality_measure')
        
        algorithms = ['hierarchical', 'kmeans', 'spectral']
        for algo in algorithms:
            if f'train_{algo}_error' in df.columns:
                comparison_cols.append(f'train_{algo}_error')
            if f'test_{algo}_error' in df.columns:
                comparison_cols.append(f'test_{algo}_error')
        
        additional_datasets = ['hopkins12']
        for dataset in additional_datasets:
            for algo in algorithms:
                col = f'{dataset}_{algo}_error'
                if col in df.columns:
                    comparison_cols.append(col)
        
        comparison_df = df[comparison_cols].copy()
        error_cols = [col for col in comparison_cols if 'error' in col]
        
        for col in error_cols:
            if col in comparison_df.columns:
                comparison_df[col] = comparison_df[col].apply(
                    lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A"
                )
                comparison_df[col] = comparison_df[col].replace('N/A', pd.NA)

        if 'orthogonality_measure' in comparison_df.columns:
            comparison_df['orthogonality_measure'] = comparison_df['orthogonality_measure'].apply(
                lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"
            )

        return comparison_df

    
    def create_heatmap_visualization(self, df, save_path = 'out/model_heatmap.png'):
        error_cols = [col for col in df.columns if 'error' in col and col in df.columns]
        
        if not error_cols:
            print("No error columns found for heatmap")
            return
        
        heatmap_data = df[['model_name'] + error_cols].copy()
        
        for col in error_cols:
            if heatmap_data[col].dtype == 'object':
                heatmap_data[col] = heatmap_data[col].str.replace('%', '').str.replace('N/A', 'nan')
                heatmap_data[col] = pd.to_numeric(heatmap_data[col], errors='coerce')
        
        heatmap_data = heatmap_data.set_index('model_name')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn_r', fmt='.2f', 
                   cbar_kws={'label': 'Error Rate (%)'})
        plt.title('Model Performance Heatmap')
        plt.xlabel('Evaluation Metrics')
        plt.ylabel('Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Heatmap saved to {save_path}")
    
    def shorten_model_name(self, model_name):
        config = self.parse_model_name(model_name)
        
        parts = []
        
        if config['dataset'] == 'hopk155':
            parts.append('H155')
        elif config['dataset'] == 'kt':
            parts.append('KT')
        
        config_parts = []
        if config['orthogonality'] == 'ortho':
            config_parts.append('O')
        if not config['feature_diff_excluded']:
            config_parts.append('F')
        if config['alpha_zero']:
            config_parts.append('alpha0')
        
        if config_parts:
            parts.append(''.join(config_parts))
        else:
            parts.append('base')
        
        if config['trained_on_full']:
            parts.append('full')
        else:
            parts.append('split')
        
        return '_'.join(parts)

    def create_typst_table(self, df, save_path = 'out/model_table.typ', include_orthogonality=True):
        display_df = df.copy()
        
        dataset_name = display_df['dataset'].iloc[0].upper()
        pre_epochs = display_df['pretraining_epochs'].iloc[0]
        full_epochs = display_df['full_epochs'].iloc[0]
        
        has_hopkins12 = False
        if 'hopkins12_hierarchical_error' in df.columns:
            hopkins12_col = df['hopkins12_hierarchical_error']
            has_hopkins12 = (hopkins12_col.notna() & (hopkins12_col != 'N/A')).any()
        
        has_orthogonality = include_orthogonality and 'orthogonality_measure' in df.columns
        
        base_cols = 3
        total_cols = base_cols
        if has_hopkins12:
            total_cols += 1
        if has_orthogonality:
            total_cols += 1
        
        typst_code = """#set table(
    stroke: (x, y) => if y == 2 {
        (bottom: 1pt + black)
    },
    align: (x, y) => (
        if x > 0 { center }
        else { left }
    )
)

#set text(
    font: "New Computer Modern",
    size: 10pt
)

#figure([
#table(
    columns: """ + str(total_cols) + """,
    stroke: none,
    table.cell(colspan: """ + str(total_cols) + """)[#align(center)[*""" + dataset_name + """ Model Comparison*]],
    table.hline(stroke: 0.5pt),
    [*Configuration*], [*Train*], [*Test*],"""
        
        if has_hopkins12:
            typst_code += " [*Hopkins12*],"
        if has_orthogonality:
            typst_code += " [*Orthogonality*],"
        
        typst_code += """
    table.hline(stroke: 0.5pt),
"""
    
        for _, row in display_df.iterrows():
            config = self.parse_model_name(row['model_name'])
            config_desc = []
            if config['orthogonality'] == 'ortho':
                config_desc.append('Orthogonal')
            if not config['feature_diff_excluded']:
                config_desc.append('FeatDiff')
            if config['alpha_zero']:
                config_desc.append('$alpha=0$')
        
            config_str = ', '.join(config_desc) if config_desc else 'Baseline'
            
            train_error = row.get('train_hierarchical_error', 'N/A')
            test_error = row.get('test_hierarchical_error', 'N/A')
            
            def format_compact_error(error_val):
                if pd.isna(error_val) or error_val == 'N/A':
                    return '-'
                elif isinstance(error_val, str) and '%' in error_val:
                    return error_val
                else:
                    return f'{error_val:.2%}'
            
            train_formatted = format_compact_error(train_error)
            test_formatted = format_compact_error(test_error)
            
            row_line = f"    [{config_str}], [{train_formatted}], [{test_formatted}]"
            
            if has_hopkins12:
                hopkins12_error = row.get('hopkins12_hierarchical_error', 'N/A')
                hopkins12_formatted = format_compact_error(hopkins12_error)
                row_line += f", [{hopkins12_formatted}]"
            
            if has_orthogonality:
                ortho_measure = row.get('orthogonality_measure', 'N/A')
                if pd.isna(ortho_measure) or ortho_measure == 'N/A':
                    ortho_formatted = '-'
                else:
                    ortho_formatted = ortho_measure
                row_line += f", [{ortho_formatted}]"
            
            typst_code += row_line + ",\n"
        
        typst_code += """    table.hline(stroke: 0.5pt),
    table.cell(colspan: """ + str(total_cols) + """)[#align(center)[""" + str(pre_epochs) + """ pre-training epochs, """ + str(full_epochs) + """ full training epochs]],
    table.cell(colspan: """ + str(total_cols) + """)[#align(center)[evaluation on Hierarchical Clustering with Mean Clustering Error"""
        
        if has_orthogonality:
            typst_code += " + Basis Function Orthogonality"
        
        typst_code += """]],
),
]) <model-config-table>
"""

        with open(save_path, 'w') as f:
            f.write(typst_code)
    
        print(f"Typst table saved to {save_path}")
        return typst_code

    def create_tables(self, df, include_orthogonality=True):
        unique_base_training_configs = df[['dataset', 'pretraining_epochs', 'full_epochs']].drop_duplicates()
        
        for _, config_row in unique_base_training_configs.iterrows():
            dataset = config_row['dataset']
            pre_epochs = config_row['pretraining_epochs'] 
            full_epochs = config_row['full_epochs']
            
            filtered_df = df[
                (df['dataset'] == dataset) &
                (df['pretraining_epochs'] == pre_epochs) &
                (df['full_epochs'] == full_epochs)
            ]
            
            if filtered_df.empty:
                continue
            
            save_path = f'out/model_table_{dataset}_{pre_epochs}_{full_epochs}.typ'
            self.create_typst_table(filtered_df, save_path, include_orthogonality)
            self.create_heatmap_visualization(filtered_df, save_path.replace('.typ', '_heatmap.png'))
            

def main():
    evaluator = ModelEvaluator()
    
    models_to_evaluate = []
    results_df = evaluator.evaluate_models(include_orthogonality=True)
    comparison_table = evaluator.create_comparison_table(results_df)
    
    results_df.to_csv('out/model_evaluation_detailed.csv', index=False)
    comparison_table.to_csv('out/model_comparison_table.csv', index=False)

    evaluator.create_tables(comparison_table, include_orthogonality=True)

    return results_df, comparison_table

if __name__ == '__main__':
    main()