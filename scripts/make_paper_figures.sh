#!/bin/bash
# Copy the figures used by paper/main.tex from plots/ (regenerate them first with the scripts).
cd "$(dirname "$0")/.."
cp plots/multipoles_c000_hod239_z_losz_rand.png paper/figures/v2_multipoles_rand.png
cp plots/multipoles_c000_hod239_z_losz_data.png paper/figures/v2_multipoles_gal.png
cp plots/environment_c000_hod239_z_losz.png     paper/figures/v2_environment.png
for f in plots/derivative_*_z.png plots/fisher_*.png plots/robustness_*.png; do
    [ -f "$f" ] && cp "$f" paper/figures/
done
echo "paper/figures updated"
