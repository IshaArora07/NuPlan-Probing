Based on everything I know about your architecture, training state, and the scores you’re seeing, here is a prioritized list organized by impact and implementation effort.

Tier 1 — High impact, low effort, do these first:
1. Train longer
Your single biggest lever. At epoch 5 you’re nowhere near convergence. The EMoE paper trained 35 epochs on 6x more data. Getting to epoch 15-20 will improve trajectory quality, comfort, and overall score more than any architectural change. No code changes needed.
2. Add more training data
If you can expand from 194K to 400-500K scenarios, the diversity improvement will directly help the router specialize and the experts generalize to edge cases seen in simulation. More data also stabilizes the cls_loss rise we observed.
3. Fix comfort by passing velocity to the planner
As discussed, your inference block discards predicted velocity. Pass speed as a 4th channel in candidate_trajectories so the LQR tracker has explicit speed targets rather than inferring them from position differences. This directly addresses your 0.666 comfort score. One code change in the inference block.
4. Set learning_based_score_weight=0.0 temporarily
At epoch 5 your pi_head scores are unreliable due to rising cls_loss. The model probability is potentially hurting trajectory selection. Setting this to 0.0 relies purely on rule-based scoring which is already giving you perfect collision, drivable area, TTC, and progress scores.

Tier 2 — Medium impact, medium effort:
5. Add reference line conditioning to mode queries
Currently your scene-specific queries use only KMeans anchor endpoints and scene type embeddings. Adding the nearest reference line direction as an additional conditioning signal would give the decoder geometric road-following guidance similar to what PLUTO gets from its reference-line-conditioned decoder. This would significantly improve trajectory quality in curved road scenarios.
6. Increase number of candidates via anchor diversity
You currently have Ka=6 candidates from a single expert. PLUTO has 18-30 from reference line × mode combinations. One way to increase diversity without reference lines is to use multiple anchor sets — for example, generate 3 anchor variants per scene type (aggressive, normal, conservative speed profiles) giving you 18 candidates. This improves the trajectory evaluator’s ability to find a good trajectory in any situation.
7. Apply trajectory smoothing post-processing
Add a simple post-processing step after the inference block that applies Savitzky-Golay or cubic spline smoothing along the time dimension of candidate_trajectories. This directly reduces jerk and improves comfort scores without retraining. Implementable entirely in pluto_planner.py.
8. Reduce w_cls to 0.3 and add label smoothing
As discussed earlier — stabilizes the cls_loss rise and improves mode scoring quality over subsequent epochs. Already prepared in our earlier code.

Tier 3 — High impact but requires significant implementation:
9. Add reference line aware decoding
The most impactful architectural change. Build reference lines into your mode query generator — instead of purely KMeans anchors, condition each mode query on both the scene type anchor AND the nearest reference line waypoints. This is essentially bridging EMoE’s scene-adaptive routing with PLUTO’s reference-line guidance, giving you the best of both.
10. Multi-modal velocity prediction in InteractionPredDecoder
Currently predicting only (x, y) for agents. Extending to (x, y, heading) or (x, y, cos_yaw, sin_yaw) improves agent prediction quality which directly helps collision avoidance scoring in complex interactive scenarios. The _get_agent_info shape[-1] == 3 branch already handles 3-channel predictions.
11. Teacher forcing for first 5 epochs then switch to predicted routing
Now that you have the infrastructure built, enabling teacher forcing for a fresh training run would improve expert specialization in early epochs, leading to better trajectory quality faster.
12. Contrastive learning from PLUTO
PLUTO’s original codebase supports contrastive learning between positive and negative trajectory samples. Re-enabling this for your EMoE model would improve the quality of the encoder representations that feed into the router and decoder, particularly for distinguishing between similar-looking scenes that require different maneuvers.

Tier 4 — Research-level enhancements for later:
13. Soft routing with temperature annealing
Instead of hard argmax routing, use soft routing with a temperature parameter that starts high (soft mixture) and anneals to hard routing over training. This avoids the winner-take-all instability in early training that contributes to the rising cls_loss.
14. Auxiliary scene classification loss on encoder output
Add a lightweight scene classification head directly on the ego encoder token, supervised with your scene labels, as an auxiliary loss alongside the router CE loss. This encourages the encoder to learn scene-discriminative representations that make routing easier and expert specialization more pronounced.
15. Diffusion-based trajectory refinement
After the planning decoder produces initial trajectory candidates, apply a lightweight diffusion refinement step conditioned on the scene tokens. This post-processes the raw predicted waypoints into smoother, more physically plausible trajectories. Research-level but has shown strong results in recent motion planning papers.

My recommended order of execution given your current state:
First continue training to epoch 15 with the cls_loss fixes already applied. Then implement the velocity fix for comfort and the trajectory smoothing in the planner. Then evaluate scores again before deciding whether to invest in the architectural changes. The architectural changes in Tier 3 are most valuable if you’re starting a fresh training run — retrofitting them onto a partially trained checkpoint is harder.​​​​​​​​​​​​​​​​
