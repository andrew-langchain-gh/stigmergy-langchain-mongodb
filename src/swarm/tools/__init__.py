"""Tools are built per agent, not shared globally.

Which tools an agent holds *is* its permission model. Phase-2 roles in particular get a
narrow, typed set of allowed operations rather than free-form document rewriting — without
that, "negotiation by editing" degrades into agents silently clobbering each other's
reasoning, which is exactly the failure mode the structural-conflict premise avoids.
"""
