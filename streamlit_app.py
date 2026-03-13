import streamlit as st
from pathlib import Path
import runpy

home = Path(__file__).parent / "00_Home.py"
runpy.run_path(str(home))