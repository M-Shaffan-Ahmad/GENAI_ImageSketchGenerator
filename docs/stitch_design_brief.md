> Update: the supplied Stitch export has now been implemented. See
> `report/stitch/provenance.json` and the updated app screenshots. This brief is
> a prepared prompt, not a verified record of the export's original generation prompt.

# Copyable Google Stitch design brief

Create an academic browser application called **Generative AI Image Restoration
and Sketch Generator**, for a CS4065 assignment using flower/object photographs.
Use a clean light interface, dark slate text, emerald primary actions, white cards
with subtle borders, and readable spacing. It must work at desktop width and a
390px mobile width without horizontal overflow.

Provide four visibly named workspaces:

1. Universal Restoration
2. Hard-Routed Restoration
3. Soft Mixture-of-Experts Restoration
4. Object-to-Sketch Generator

On desktop, use workspace tabs above a two-column layout: a roughly 300px control
card and a flexible output card. On mobile, stack controls and outputs vertically
and wrap workspace navigation. Do not hide the main actions behind menus.

All workspaces have an image file upload and an alternative clean flower sample
selector. Inputs are mutually exclusive. Show a disabled primary button until an
input is chosen, a loading state during inference, and readable error messages.

The three restoration workspaces let the user choose: use image as uploaded,
salt-and-pepper noise, Gaussian blur, or rectangular occlusion. Only show settings
for the selected corruption. Noise probability ranges from .02 to .15; blur has
kernel choices 3/5/7 and sigma .5 to 2.5; occlusion has covered area .10 to .35 and
1/2/3 rectangles. The main action is **Restore image**.

Restoration outputs show input and restored image side by side. If a clean reference
exists because a sample or simulated corruption was used, also show clean reference
and absolute error. If the uploaded image has no known reference, do not invent one.
Show inference time and a **Download result** PNG action.

Hard routing also shows the predicted condition, four labeled classifier probability
bars and whether the clean identity bypass was chosen. Soft MoE shows four labeled
contribution bars: identity/clean, noise, blur, occlusion. Describe these as learned
mixture weights; do not call the soft output a single selected expert.

Sketch generation has a three-choice style selector: **Fine Pencil**, **Technical
Ink**, **Tonal Charcoal**. Hide corruption controls in this workspace. Its main action
is **Generate sketch**. Results show photograph and generated sketch, the selected
style name, time and PNG download. Do not show a photo-versus-sketch error map.

Generate desktop and mobile screens for all four workspaces, including empty,
loading, successful and error states. Use flower/object imagery. Use semantic
labels, adequate contrast, keyboard-visible focus and clear buttons. Do not add
unrelated login, subscriptions, chat, model configuration or deployment controls.

---

This brief is prepared input, **not evidence that Stitch was used**. Save the actual
Stitch project, prompt, exported designs and screenshots after generation. The
existing React implementation predates that step; document this chronology.
