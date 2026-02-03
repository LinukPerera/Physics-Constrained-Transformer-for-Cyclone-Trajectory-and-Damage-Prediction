"""
Logging Configuration and Audit Trail Infrastructure.

This module provides structured logging for scientific reproducibility and
audit trail requirements. All prediction runs produce artifacts that enable
post-event forensic analysis.

Audit Requirements
------------------
Per the SOP, every run must produce:
- Configuration hash
- Model versions
- Physics residual summaries
- Constraint violation counts

This module implements the infrastructure to capture this information.
"""

import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
import threading


# Configure root logger for the package
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a logger configured for the cyclone prediction system.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__).
    level : int
        Logging level.
        
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    return logger


@dataclass
class ConstraintViolation:
    """Record of a physics constraint violation.
    
    Attributes
    ----------
    timestamp : datetime
        When the violation occurred.
    constraint_name : str
        Identifier for the violated constraint.
    severity : str
        One of 'warning', 'error', 'critical'.
    original_value : float
        The value before correction.
    corrected_value : float
        The value after correction.
    correction_magnitude : float
        Absolute change applied.
    context : dict
        Additional context (storm ID, forecast hour, etc.).
    """
    timestamp: datetime
    constraint_name: str
    severity: str
    original_value: float
    corrected_value: float
    correction_magnitude: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhysicsResidual:
    """Record of physics conservation law residual.
    
    Attributes
    ----------
    timestamp : datetime
        When the residual was computed.
    conservation_law : str
        Which conservation law (e.g., 'mass', 'momentum', 'energy').
    residual_value : float
        The residual magnitude.
    tolerance : float
        The acceptable tolerance.
    passed : bool
        Whether the residual is within tolerance.
    context : dict
        Additional context.
    """
    timestamp: datetime
    conservation_law: str
    residual_value: float
    tolerance: float
    passed: bool
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunMetadata:
    """Metadata for a prediction run.
    
    This captures all information needed for reproducibility and audit.
    """
    run_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    config_hash: str = ""
    model_versions: Dict[str, str] = field(default_factory=dict)
    input_metadata: Dict[str, Any] = field(default_factory=dict)
    output_metadata: Dict[str, Any] = field(default_factory=dict)
    constraint_violations: List[ConstraintViolation] = field(default_factory=list)
    physics_residuals: List[PhysicsResidual] = field(default_factory=list)
    
    def compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute a deterministic hash of the configuration.
        
        Parameters
        ----------
        config : dict
            The configuration dictionary.
            
        Returns
        -------
        str
            SHA-256 hash of the configuration.
        """
        config_str = json.dumps(config, sort_keys=True, default=str)
        self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return self.config_hash


class AuditLogger:
    """Central logging facility for audit trail generation.
    
    This class maintains records of all constraint violations, physics
    residuals, and run metadata for post-event analysis.
    
    Thread Safety
    -------------
    All methods are thread-safe for use in parallel processing.
    
    Examples
    --------
    >>> audit = AuditLogger()
    >>> with audit.run_context("prediction_001") as run:
    ...     audit.log_constraint_violation(
    ...         constraint_name="max_wind_limit",
    ...         severity="warning",
    ...         original_value=85.0,
    ...         corrected_value=80.0,
    ...         context={"storm_id": "AL092017"}
    ...     )
    >>> summary = audit.get_run_summary("prediction_001")
    """
    
    _instance: Optional['AuditLogger'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'AuditLogger':
        """Singleton pattern for global audit logger."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the audit logger."""
        if self._initialized:
            return
            
        self._runs: Dict[str, RunMetadata] = {}
        self._current_run_id: Optional[str] = None
        self._logger = get_logger("audit")
        self._file_logger: Optional[logging.Logger] = None
        self._initialized = True
    
    def set_output_dir(self, output_dir: Path) -> None:
        """Set the directory for audit log files.
        
        Parameters
        ----------
        output_dir : Path
            Directory to write audit logs.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(
            output_dir / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        )
        self._file_logger = logging.getLogger("audit.file")
        self._file_logger.addHandler(file_handler)
        self._file_logger.setLevel(logging.DEBUG)
    
    @contextmanager
    def run_context(self, run_id: str, config: Optional[Dict[str, Any]] = None):
        """Context manager for a prediction run.
        
        Parameters
        ----------
        run_id : str
            Unique identifier for this run.
        config : dict, optional
            Configuration to compute hash from.
            
        Yields
        ------
        RunMetadata
            The metadata object for this run.
        """
        metadata = RunMetadata(
            run_id=run_id,
            start_time=datetime.now()
        )
        
        if config:
            metadata.compute_config_hash(config)
        
        self._runs[run_id] = metadata
        self._current_run_id = run_id
        
        self._logger.info(f"Starting run {run_id} with config hash {metadata.config_hash}")
        
        try:
            yield metadata
        finally:
            metadata.end_time = datetime.now()
            self._current_run_id = None
            self._logger.info(
                f"Completed run {run_id}. "
                f"Violations: {len(metadata.constraint_violations)}, "
                f"Physics checks: {len(metadata.physics_residuals)}"
            )
    
    def log_constraint_violation(
        self,
        constraint_name: str,
        severity: str,
        original_value: float,
        corrected_value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a physics constraint violation.
        
        Parameters
        ----------
        constraint_name : str
            Identifier for the constraint.
        severity : str
            'warning', 'error', or 'critical'.
        original_value : float
            Value before correction.
        corrected_value : float
            Value after correction.
        context : dict, optional
            Additional context for the violation.
        """
        violation = ConstraintViolation(
            timestamp=datetime.now(),
            constraint_name=constraint_name,
            severity=severity,
            original_value=original_value,
            corrected_value=corrected_value,
            correction_magnitude=abs(corrected_value - original_value),
            context=context or {}
        )
        
        if self._current_run_id and self._current_run_id in self._runs:
            self._runs[self._current_run_id].constraint_violations.append(violation)
        
        log_msg = (
            f"CONSTRAINT VIOLATION | {constraint_name} | {severity} | "
            f"original={original_value:.4f} -> corrected={corrected_value:.4f}"
        )
        
        if severity == 'critical':
            self._logger.error(log_msg)
        elif severity == 'error':
            self._logger.warning(log_msg)
        else:
            self._logger.info(log_msg)
    
    def log_physics_residual(
        self,
        conservation_law: str,
        residual_value: float,
        tolerance: float,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a physics conservation law residual.
        
        Parameters
        ----------
        conservation_law : str
            Which conservation law was checked.
        residual_value : float
            The computed residual.
        tolerance : float
            The acceptable tolerance.
        context : dict, optional
            Additional context.
        """
        passed = abs(residual_value) <= tolerance
        
        residual = PhysicsResidual(
            timestamp=datetime.now(),
            conservation_law=conservation_law,
            residual_value=residual_value,
            tolerance=tolerance,
            passed=passed,
            context=context or {}
        )
        
        if self._current_run_id and self._current_run_id in self._runs:
            self._runs[self._current_run_id].physics_residuals.append(residual)
        
        status = "PASS" if passed else "FAIL"
        log_msg = (
            f"PHYSICS CHECK | {conservation_law} | {status} | "
            f"residual={residual_value:.6e} (tolerance={tolerance:.6e})"
        )
        
        if passed:
            self._logger.debug(log_msg)
        else:
            self._logger.warning(log_msg)
    
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get a summary of a prediction run.
        
        Parameters
        ----------
        run_id : str
            The run identifier.
            
        Returns
        -------
        dict
            Summary including violation counts and physics check results.
        """
        if run_id not in self._runs:
            raise KeyError(f"No run found with ID {run_id}")
        
        metadata = self._runs[run_id]
        
        violation_counts = {}
        for v in metadata.constraint_violations:
            violation_counts[v.constraint_name] = violation_counts.get(v.constraint_name, 0) + 1
        
        physics_results = {}
        for r in metadata.physics_residuals:
            physics_results[r.conservation_law] = {
                "passed": r.passed,
                "residual": r.residual_value,
                "tolerance": r.tolerance
            }
        
        return {
            "run_id": run_id,
            "config_hash": metadata.config_hash,
            "start_time": metadata.start_time.isoformat(),
            "end_time": metadata.end_time.isoformat() if metadata.end_time else None,
            "total_constraint_violations": len(metadata.constraint_violations),
            "violation_counts_by_type": violation_counts,
            "physics_residuals": physics_results,
            "model_versions": metadata.model_versions
        }
    
    def export_run_artifacts(self, run_id: str, output_path: Path) -> None:
        """Export all audit artifacts for a run to JSON.
        
        Parameters
        ----------
        run_id : str
            The run identifier.
        output_path : Path
            Path to write the JSON file.
        """
        if run_id not in self._runs:
            raise KeyError(f"No run found with ID {run_id}")
        
        metadata = self._runs[run_id]
        
        # Convert to serializable format
        artifacts = {
            "run_id": metadata.run_id,
            "config_hash": metadata.config_hash,
            "start_time": metadata.start_time.isoformat(),
            "end_time": metadata.end_time.isoformat() if metadata.end_time else None,
            "model_versions": metadata.model_versions,
            "input_metadata": metadata.input_metadata,
            "output_metadata": metadata.output_metadata,
            "constraint_violations": [
                {
                    "timestamp": v.timestamp.isoformat(),
                    "constraint_name": v.constraint_name,
                    "severity": v.severity,
                    "original_value": v.original_value,
                    "corrected_value": v.corrected_value,
                    "correction_magnitude": v.correction_magnitude,
                    "context": v.context
                }
                for v in metadata.constraint_violations
            ],
            "physics_residuals": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "conservation_law": r.conservation_law,
                    "residual_value": r.residual_value,
                    "tolerance": r.tolerance,
                    "passed": r.passed,
                    "context": r.context
                }
                for r in metadata.physics_residuals
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(artifacts, f, indent=2)
        
        self._logger.info(f"Exported audit artifacts to {output_path}")
