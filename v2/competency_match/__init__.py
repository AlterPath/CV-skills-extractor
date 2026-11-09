"""Анализ соответствия резюме матрице компетенций Альянса (версия 2)."""
from .assess import Analysis, CompetencyResult, RoleResult, analyze
from .matrix import Matrix, load_matrix
from .text import read_resume

__all__ = ["Analysis", "CompetencyResult", "Matrix", "RoleResult", "analyze", "load_matrix", "read_resume"]
