* **Coverage.** __S_TRACED__ survey patches were traced in the 21 eligible scrolls; __S_OK__ pass the sheet check and
  __S_INKED__ were inked, in 20 scrolls. No PHerc. 175B trace passed the sheet check (__PHerc0175B__) and few PHerc. 268
  traces did (__PHerc0268__). Another __F_TRACED__ patches were traced to follow up leads, plus one PHerc. 1545 patch
  from the Windows end-to-end test.
* **What known text looks like, and its limits.** On PHerc. 343 the known-text segment and an overlapping trace a few
  voxels inward score high on both ink scores, with the signal on the inner face. An overlapping trace a few voxels
  outward, a patch a few windings away and a patch elsewhere score flat. A flat card means "nothing found here",
  not "nothing in this scroll", and a trace over real text can still score flat.
* **Most survey patches look like blank papyrus to these models.** Survey medians: face asymmetry __S_ASYM_MED__
  (known text 3.2 and 4.0), face consistency __S_CONS_MED__ (known text 0.52 and 0.77).
* **One survey patch scores like the known text on both ink scores: PHerc. 846A at (2765, 2952, 4205).** Its signal
  is on the inner face. Every model pair with seed43 and Hecate prefers that face too, two of them only weakly (1.3x and
  1.4x, against 3.2x to 4.4x on the known text). 15 of 17 patches traced around it put their signal on the inner
  face. Its strongest follow-up, 30 voxels outward, has a higher face asymmetry than the known text (5.55;
  consistency 0.64), all six pairs of four models prefer its inner face (2.6x to 8.9x), and a 17-depth sweep there
  peaks at the sheet surface. The caveat: on that patch a quarter of the agreed ink sits on the darkest tenth of the
  CT (known text: a tenth), so cracks and gaps may contribute. **No letters can be read yet.**
* **Other leads.** PHerc. 846B: 10 of 14 follow-ups inner (3 outer, 1 tie); its side calls are uncertain on flat
  sheet and its patches lean on dark CT. PHerc. 1545 at (3499, 4074, 9431), the patch made by the Windows end-to-end
  test, has the most even four-model agreement of any lead (all six pairs 2.3x to 5.1x), but the windings around it
  are mixed (4 inner, 1 outer, 1 tie; median consistency 0.19). PHerc. 211, the first lead of this project, was
  dropped: its neighbouring windings flip sides and seed43 does not confirm it.
* **What to do with this.** The leads table and the meshes in `results/meshes/` are starting points for careful
  segmentation and First Letters attempts, PHerc. 846A first. The same card, run on any new trace, shows once the two
  ink models have run (~6 minutes on a laptop GPU) whether a patch is worth a closer look.
