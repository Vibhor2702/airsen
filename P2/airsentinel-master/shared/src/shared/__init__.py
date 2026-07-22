"""
Integration/fusion layer that combines AirSentinel's independent tracks into one zone-keyed
view — the "03 · FUSION LAYER" and "04 · OUTPUT LAYER" from the design plan's system
architecture (slide 5), scaffolded ahead of full satellite/enforcement integration so wiring
in the remaining pieces is a data drop, not a rewrite.

Every function here is deterministic (formula/template-based) — nothing is fitted to data,
so nothing here can overfit. Where a real external feed isn't available yet (satellite
attribution, SAFAR/Supersite comparison), the corresponding column is reported as missing
rather than filled with a placeholder number — see fuse.py.
"""

__version__ = "0.1.0"
