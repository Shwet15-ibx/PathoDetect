"""
MLOps Tracking for PathoDetect+
Simplified version for experiment tracking
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

class MLflowTracker:
    def __init__(self, config):
        self.config = config
        self.experiment_name = config['mlops']['tracking']['experiment_name']
        self.run_name = config['mlops']['tracking']['run_name']
        self.log_dir = "./mlruns"
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize tracking
        self.current_run: Optional[Dict[str, Any]] = None
        self.metrics = {}
        self.artifacts = {}
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """Start a new MLflow run"""
        if run_name is None:
            run_name = self.run_name
        
        self.current_run = {
            'run_id': f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'run_name': run_name,
            'start_time': datetime.now().isoformat(),
            'metrics': {},
            'artifacts': {},
            'params': {}
        }
        
        print(f"Started MLflow run: {self.current_run['run_id']}")
        return self.current_run['run_id']
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics for the current run"""
        if self.current_run is None:
            self.start_run()
        
        if self.current_run is not None:
            for key, value in metrics.items():
                self.current_run['metrics'][key] = value
        
        print(f"Logged metrics: {list(metrics.keys())}")
    
    def log_artifacts(self, artifacts: Dict[str, Any]) -> None:
        """Log artifacts for the current run"""
        if self.current_run is None:
            self.start_run()
        
        if self.current_run is not None:
            for key, value in artifacts.items():
                self.current_run['artifacts'][key] = value
        
        print(f"Logged artifacts: {list(artifacts.keys())}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters for the current run"""
        if self.current_run is None:
            self.start_run()
        
        if self.current_run is not None:
            for key, value in params.items():
                self.current_run['params'][key] = value
        
        print(f"Logged parameters: {list(params.keys())}")
    
    def end_run(self) -> None:
        """End the current run and save to disk"""
        if self.current_run is None:
            return
        
        self.current_run['end_time'] = datetime.now().isoformat()
        
        # Save run to file
        run_file = os.path.join(self.log_dir, f"{self.current_run['run_id']}.json")
        with open(run_file, 'w') as f:
            json.dump(self.current_run, f, indent=2, default=str)
        
        print(f"Ended MLflow run: {self.current_run['run_id']}")
        self.current_run = None
    
    def get_runs(self) -> List[Dict]:
        """Get all previous runs"""
        runs = []
        for filename in os.listdir(self.log_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.log_dir, filename), 'r') as f:
                    run_data = json.load(f)
                    runs.append(run_data)
        return runs
    
    def get_best_run(self, metric: str = 'f1_score') -> Optional[Dict]:
        """Get the best run based on a metric"""
        runs = self.get_runs()
        if not runs:
            return None
        
        best_run = max(runs, key=lambda x: x.get('metrics', {}).get(metric, 0))
        return best_run
    
    def compare_runs(self, metric: str = 'f1_score') -> pd.DataFrame:
        """Compare runs and return as DataFrame"""
        runs = self.get_runs()
        if not runs:
            return pd.DataFrame()
        
        data = []
        for run in runs:
            row = {
                'run_id': run['run_id'],
                'run_name': run['run_name'],
                'start_time': run['start_time'],
                'metric': run.get('metrics', {}).get(metric, 0)
            }
            data.append(row)
        
        return pd.DataFrame(data) 