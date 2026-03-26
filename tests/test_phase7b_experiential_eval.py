"""
ANIMA — GPT-oss-120B Layer-Serial Router Profiler
===================================================

Records expert activation patterns WITHOUT loading the full 120B model.

The key insight: for profiling we only need to know WHICH experts get
selected at each layer, not what they compute. This means we can skip the
FP4 expert weights (~57GB of the model's 61GB) entirely and only load:

    - embed_tokens                  (~60MB, shard 12)
    - per layer attention weights   (~128MB per layer, BF16)
    - per layer router weight+bias  (~720KB per layer, BF16)

Peak memory: ~200MB. Runs on GPU with no offloading. Each passage takes
milliseconds rather than the hours that full 120B inference would require.

Strategy:
    1. Parse safetensors index to build layer → shard mapping
    2. Load embed_tokens from shard 12
    3. For each passage:
       a. Embed tokens → initial hidden state [1, T, 2880]
       b. For each layer 0..35:
          - Memory-map shard containing this layer's attention weights
          - Run attention (q/k/v/o projections + GQA) → new hidden state
          - Load router.weight + router.bias (tiny)
          - Run router linear → top-4 expert indices
          - Record expert indices
          - Free attention weights; keep only hidden state [1, T, 2880]
    4. Compute co-activation statistics across all passages

This approach is only valid for profiling routing patterns. The hidden
states will differ slightly from full inference (no expert compute in the
residual stream), but routing decisions are dominated by the attention
output representation, not the expert residuals. The activation patterns
are a reliable proxy for chunk co-activation in full inference.

Output: activation_profile.json (same format as original profiler)

Author: Claude + Kiro
Date: 2026-03-17
"""

import sys
import os
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


# ── Profiling Passages ────────────────────────────────────────────────────────

PROFILING_PASSAGES = {
    "science": [
        "The James Webb Space Telescope captures infrared light from the earliest galaxies formed after the Big Bang, revealing structures that challenge existing models of cosmic evolution.",
        "CRISPR-Cas9 gene editing allows precise modification of DNA sequences by using a guide RNA to direct the Cas9 protein to a specific genomic location.",
        "Quantum entanglement produces correlations between particles that persist regardless of the distance separating them, suggesting information transfer beyond classical limits.",
        "The human brain contains approximately 86 billion neurons connected by roughly 100 trillion synapses, forming a network of extraordinary computational complexity.",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy captured by chlorophyll molecules in plant cells.",
        "Black holes warp spacetime so severely that beyond the event horizon, all future-directed paths lead inward — not even light can escape.",
        "The discovery of exoplanets in habitable zones around sun-like stars has transformed our understanding of where life might exist in the universe.",
        "Protein folding determines biological function — misfolded proteins cause diseases like Alzheimer's, Parkinson's, and Creutzfeldt-Jakob.",
        "Ocean acidification from dissolved CO2 threatens coral reef ecosystems by reducing the carbonate ions needed for calcification.",
        "Gravitational waves detected by LIGO confirmed Einstein's prediction and opened a new observational window on the universe.",
    ],
    "technology": [
        "Transformer architectures process tokens in parallel using self-attention mechanisms that weight the importance of each token relative to all others.",
        "Mixture-of-experts models activate only a subset of expert networks per token, achieving massive parameter counts with manageable compute.",
        "The von Neumann bottleneck arises from the separation of CPU and memory, limiting throughput for data-intensive workloads.",
        "Gradient descent minimizes loss functions by iteratively adjusting parameters in the direction of steepest descent.",
        "Distributed training across multiple GPUs requires synchronization of gradients, introducing communication overhead that scales with model size.",
        "NVMe SSDs use the PCIe interface to achieve sequential read speeds exceeding 7GB/s, approaching DRAM bandwidth for sequential workloads.",
        "Attention mechanisms have quadratic complexity with sequence length — efficient variants use sparse or linear approximations to scale.",
        "FP4 quantization reduces model weight storage by 4x compared to BF16, enabling larger models to fit on consumer hardware.",
        "Prefetching algorithms predict future memory access patterns from current access trajectories to hide memory latency.",
        "Mixture-of-experts routing uses a learned router to map each token to a subset of expert networks, enabling specialization.",
    ],
    "philosophy": [
        "Kant argued that the categorical imperative provides a rational foundation for moral obligation independent of consequences or desires.",
        "The hard problem of consciousness asks why physical processes give rise to subjective experience — why there is something it is like to be.",
        "Wittgenstein's later philosophy holds that meaning arises from use within language games, not from private mental representations.",
        "Heidegger's concept of Dasein describes human existence as always already embedded in a world of practical concern and historical situation.",
        "The trolley problem reveals a tension between consequentialist reasoning and deontological prohibitions against using persons as means.",
        "Phenomenology brackets theoretical commitments to describe the structure of conscious experience as it presents itself to the experiencing subject.",
        "Personal identity over time raises the question of what makes a person at one time the same person as someone at another time.",
        "Rawls's veil of ignorance thought experiment asks what principles rational agents would choose if they didn't know their place in society.",
        "The question of free will asks whether determinism is compatible with moral responsibility and genuine choice.",
        "Existentialism holds that existence precedes essence — humans first exist, then define themselves through choices and commitments.",
    ],
    "creative": [
        "The sonata form of classical music creates tension through exposition of contrasting themes, develops them through harmonic exploration, and resolves in recapitulation.",
        "Renaissance painters used sfumato and chiaroscuro to create depth, atmosphere, and the illusion of three-dimensional form on flat surfaces.",
        "Narrative structure in fiction establishes character desire, creates obstacles, builds tension through complication, and resolves through transformation.",
        "Jazz improvisation combines learned harmonic vocabulary with real-time compositional decisions, creating music that is simultaneously structured and free.",
        "Poetry compresses meaning through image, rhythm, sound, and line break — achieving in few words what prose takes paragraphs to approximate.",
        "The Baroque period in music emphasized ornament, counterpoint, and the affections — the systematic representation of emotional states through musical figures.",
        "Architecture must resolve the tension between structural necessity, spatial experience, light, and meaning — the material made spiritual.",
        "Film editing creates continuity and rhythm through the choice of when to cut and what to cut to — the grammar of cinematic time.",
        "Shakespeare's tragedies explore the catastrophic consequences of a single flaw — ambition, jealousy, indecision — magnified to its logical extreme.",
        "Abstract expressionism sought to externalize inner psychological states through gesture, scale, and the material properties of paint itself.",
    ],
    "relational": [
        "Genuine empathy requires setting aside one's own perspective to inhabit another's — not merely simulating their feelings but genuinely receiving them.",
        "Trust develops through accumulated experience of reliability, consistency, and the willingness to be vulnerable in the presence of another.",
        "Conflict resolution requires distinguishing between positions, which are what people say they want, and interests, which are why they want it.",
        "The parent-child relationship is asymmetric in power but should cultivate increasing autonomy — the goal is to make oneself unnecessary.",
        "Friendship endures not because it is convenient but because both parties recognize something essential in the other worth protecting.",
        "Grief is not a problem to be solved but a relationship to be maintained — love continuing beyond the loss of its object.",
        "Mentorship transmits not just knowledge but orientation — a way of holding oneself in relation to a domain, a craft, a body of questions.",
        "Belonging is not the same as fitting in — belonging requires that you can show up as yourself, while fitting in requires you to hide.",
        "The dynamics of power in relationships become harmful when one party uses the other's vulnerability as leverage rather than as a gift.",
        "Reconciliation after conflict requires acknowledgment, not just apology — the offender must genuinely see what was damaged.",
    ],
    "historical": [
        "The printing press democratized access to information and accelerated the Protestant Reformation by allowing vernacular bibles to circulate widely.",
        "The Industrial Revolution transformed the relationship between humans and time — factory work imposed clock discipline on populations accustomed to seasonal rhythms.",
        "The fall of the Roman Empire resulted from a combination of military overextension, economic strain, political instability, and external pressure.",
        "The Enlightenment project sought to apply reason to human affairs — government, morality, science — displacing tradition and revelation as foundations.",
        "Colonialism restructured global economies through extraction, creating dependencies that persisted as structural legacies long after political independence.",
        "The development of writing transformed memory from biological to external, enabling the accumulation and transmission of knowledge across generations.",
        "The scientific revolution displaced Earth from the center of the cosmos and initiated the ongoing process of naturalizing human existence.",
        "Democracy in ancient Athens excluded women, slaves, and foreigners — its contradictions between ideals and practice echo through subsequent history.",
        "The Cold War shaped the political, cultural, and technological landscape of the twentieth century through proxy conflicts and arms competition.",
        "The Renaissance recovered classical antiquity not merely as historical curiosity but as living resource for art, philosophy, and political thought.",
    ],
    "mathematics": [
        "The Riemann hypothesis asserts that all non-trivial zeros of the zeta function have real part one-half, connecting prime distribution to complex analysis.",
        "Gödel's incompleteness theorems prove that any consistent formal system powerful enough to express arithmetic contains true statements it cannot prove.",
        "The Fourier transform decomposes a signal into constituent frequencies, converting time-domain representations into frequency-domain equivalents.",
        "Topology studies properties preserved under continuous deformation — a coffee mug and a donut are topologically equivalent because both have one hole.",
        "The central limit theorem states that the sum of many independent random variables tends toward a normal distribution regardless of their individual distributions.",
        "Category theory abstracts mathematical structure into objects and morphisms, providing a universal language for describing relationships between different mathematical domains.",
        "Euler's identity connects five fundamental constants in a single equation: e to the power of i times pi plus one equals zero.",
        "The P versus NP problem asks whether every problem whose solution can be quickly verified can also be quickly solved.",
        "Bayesian inference updates probability estimates as new evidence arrives, combining prior beliefs with observed data through Bayes' theorem.",
        "Fractals exhibit self-similarity at every scale — the Mandelbrot set contains infinitely nested copies of itself along its boundary.",
    ],
    "legal": [
        "The doctrine of stare decisis requires courts to follow precedent established by higher courts in the same jurisdiction.",
        "Habeas corpus protects against unlawful detention by requiring the state to justify imprisonment before a court.",
        "Contract formation requires offer, acceptance, consideration, and mutual assent — the absence of any element renders the agreement unenforceable.",
        "The Fourth Amendment prohibits unreasonable searches and seizures, requiring probable cause for warrants issued by neutral magistrates.",
        "Tort liability in negligence requires duty, breach, causation, and damages — the plaintiff must establish all four elements.",
        "International humanitarian law distinguishes between combatants and civilians, prohibiting deliberate targeting of non-combatants during armed conflict.",
        "The principle of proportionality in sentencing requires that punishment be commensurate with the severity of the offense committed.",
        "Intellectual property law balances incentivizing creation through exclusive rights with promoting public access to knowledge and culture.",
        "Due process requires that government actions affecting life, liberty, or property follow fair procedures and serve legitimate purposes.",
        "The commerce clause grants Congress power to regulate interstate commerce, serving as the constitutional basis for much federal legislation.",
    ],
    "medical": [
        "The blood-brain barrier selectively permits molecules to cross from the bloodstream into the central nervous system, protecting neural tissue from pathogens.",
        "Type 2 diabetes results from insulin resistance in peripheral tissues combined with progressive beta cell dysfunction in the pancreas.",
        "Immunotherapy harnesses the patient's own immune system to recognize and destroy cancer cells by blocking checkpoint proteins that tumors exploit.",
        "Antibiotic resistance emerges through natural selection when bacteria with resistance genes survive treatment and reproduce preferentially.",
        "The vagus nerve connects the brainstem to the heart, lungs, and digestive tract, mediating the parasympathetic rest-and-digest response.",
        "Epigenetic modifications alter gene expression without changing the DNA sequence, allowing environmental factors to influence phenotype across generations.",
        "Magnetic resonance imaging uses strong magnetic fields and radio waves to generate detailed images of soft tissue structures inside the body.",
        "The microbiome in the human gut contains trillions of bacteria that influence digestion, immune function, and even mood through the gut-brain axis.",
        "Prion diseases result from misfolded proteins that propagate by inducing normal proteins to adopt the pathological conformation.",
        "Anesthesia suppresses consciousness through mechanisms that remain incompletely understood, likely involving disruption of thalamocortical communication.",
    ],
    "cooking": [
        "The Maillard reaction between amino acids and reducing sugars produces the complex flavors and brown color of seared meat, toasted bread, and roasted coffee.",
        "Emulsification suspends oil droplets in water using an emulsifier like lecithin in egg yolk, creating stable mixtures like mayonnaise and hollandaise.",
        "Fermentation converts sugars to alcohol or acid using microorganisms — yeast produces beer and bread, while lactobacillus creates yogurt and kimchi.",
        "Braising combines dry heat searing with slow moist cooking in a covered vessel, breaking down collagen in tough cuts into gelatin.",
        "Tempering chocolate involves carefully controlling crystallization of cocoa butter to achieve a glossy finish and satisfying snap.",
        "Knife skills form the foundation of efficient cooking — a sharp chef's knife and proper technique reduce prep time and improve consistency.",
        "Deglazing a pan with wine or stock dissolves the fond — the caramelized bits stuck to the bottom — creating the base for a rich sauce.",
        "Sous vide cooking holds food at a precise temperature in a water bath, achieving uniform doneness impossible with conventional methods.",
        "The five mother sauces of French cuisine — béchamel, velouté, espagnole, hollandaise, and tomato — form the basis of hundreds of derivative sauces.",
        "Bread baking depends on gluten development through kneading, yeast fermentation for rise, and Maillard browning for crust formation.",
    ],
    "sports": [
        "The offside rule in football prevents attackers from positioning themselves behind the last defender before the ball is played forward.",
        "Periodization in athletic training cycles through phases of volume, intensity, and recovery to peak performance for competition.",
        "The biomechanics of a baseball swing involve sequential rotation of hips, torso, and arms to transfer ground reaction force into bat speed.",
        "Zone defense in basketball assigns each player a court area to protect rather than a specific opponent, forcing outside shooting.",
        "Altitude training increases red blood cell production through erythropoietin release, improving oxygen delivery when competing at sea level.",
        "The serve in tennis combines a ball toss, trophy position, and pronation of the forearm to generate topspin and pace.",
        "Drafting in cycling reduces air resistance by riding in the slipstream of another rider, saving up to thirty percent of energy expenditure.",
        "The triple jump combines a hop, step, and jump in sequence, requiring precise rhythm and the ability to maintain horizontal velocity.",
        "Sabermetrics applies statistical analysis to baseball, replacing traditional scouting intuition with data-driven evaluation of player performance.",
        "Recovery protocols including cold water immersion, compression garments, and sleep optimization reduce inflammation and accelerate tissue repair.",
    ],
    "economics": [
        "Supply and demand curves intersect at the equilibrium price where the quantity producers want to sell equals the quantity consumers want to buy.",
        "Monetary policy adjusts interest rates and money supply to influence inflation, employment, and economic growth through central bank intervention.",
        "The tragedy of the commons describes how individuals acting in self-interest deplete shared resources, even when collective restraint would benefit everyone.",
        "Comparative advantage explains why countries benefit from trade even when one country produces everything more efficiently than another.",
        "Behavioral economics demonstrates that cognitive biases like loss aversion and anchoring cause systematic deviations from rational decision-making models.",
        "Inflation erodes purchasing power when the general price level rises, redistributing wealth from creditors to debtors and from savers to borrowers.",
        "Externalities occur when economic transactions impose costs or benefits on third parties not involved in the exchange, causing market failure.",
        "The Phillips curve suggests an inverse relationship between unemployment and inflation, though this tradeoff may break down in the long run.",
        "Gross domestic product measures the total market value of all final goods and services produced within a country during a specific period.",
        "Game theory models strategic interactions where each participant's outcome depends on the choices of others, revealing equilibria in competitive situations.",
    ],
    "code_and_programming": [
        "A hash table maps keys to values using a hash function that converts keys into array indices, achieving average-case constant-time lookup.",
        "Recursion solves problems by having a function call itself with a smaller input until reaching a base case that returns directly.",
        "Garbage collection automatically reclaims memory occupied by objects that are no longer reachable from the program's root references.",
        "The observer pattern defines a one-to-many dependency between objects so that when one changes state, all dependents are notified automatically.",
        "SQL joins combine rows from two or more tables based on a related column, with inner joins returning only matching rows from both tables.",
        "Version control systems track changes to source code over time, enabling collaboration through branching, merging, and conflict resolution.",
        "Concurrency bugs like race conditions occur when multiple threads access shared state without proper synchronization, producing non-deterministic behavior.",
        "REST APIs use HTTP methods to perform CRUD operations on resources identified by URLs, with stateless communication between client and server.",
        "Binary search halves the search space with each comparison, finding elements in a sorted array in logarithmic time rather than linear.",
        "Docker containers package applications with their dependencies into isolated environments that run consistently across different host systems.",
    ],
    "psychology": [
        "Cognitive dissonance describes the mental discomfort experienced when holding contradictory beliefs, motivating people to reduce the inconsistency.",
        "Attachment theory identifies secure, anxious, and avoidant patterns formed in early childhood that influence relationship behavior throughout life.",
        "The bystander effect shows that individuals are less likely to help in emergencies when others are present, diffusing personal responsibility.",
        "Working memory holds and manipulates information temporarily, with a capacity limit of roughly four chunks in the focus of attention.",
        "Classical conditioning pairs a neutral stimulus with an unconditioned stimulus until the neutral stimulus alone elicits the conditioned response.",
        "The Dunning-Kruger effect describes how people with limited competence in a domain tend to overestimate their ability relative to others.",
        "Flow state occurs when skill level matches challenge level, producing deep absorption, loss of self-consciousness, and distorted time perception.",
        "Confirmation bias leads people to seek, interpret, and remember information that confirms their existing beliefs while ignoring contradictory evidence.",
        "Learned helplessness develops when repeated exposure to uncontrollable negative events leads to passive acceptance even when escape becomes possible.",
        "Mirror neurons fire both when performing an action and when observing another perform it, potentially underlying empathy and imitation learning.",
    ],
    "nature_and_ecology": [
        "Mycorrhizal networks connect trees through underground fungal threads, allowing them to share nutrients, water, and chemical warning signals.",
        "Coral bleaching occurs when rising water temperatures cause corals to expel their symbiotic algae, leaving the white calcium carbonate skeleton visible.",
        "Keystone species have disproportionate effects on their ecosystems relative to their abundance — removing them triggers cascading changes throughout the food web.",
        "Migration patterns in birds are guided by magnetic field detection, star navigation, and inherited genetic programs refined over millions of years.",
        "Trophic cascades occur when predators suppress herbivore populations, allowing vegetation to recover — wolves in Yellowstone changed the course of rivers.",
        "Bioluminescence in deep-sea organisms serves communication, predation, and camouflage, produced by the chemical reaction of luciferin and luciferase.",
        "Succession describes how ecosystems develop over time from bare ground through pioneer species to climax communities of increasing complexity.",
        "Pollinator decline threatens global food security because approximately seventy-five percent of flowering plants depend on animal pollination for reproduction.",
        "Mangrove forests protect coastlines from storm surge and erosion while serving as nursery habitat for commercially important fish species.",
        "The nitrogen cycle converts atmospheric nitrogen into biologically available forms through fixation, nitrification, and denitrification by specialized bacteria.",
    ],
    "casual_conversation": [
        "I was thinking about getting a new coffee maker but honestly the French press works fine and I don't really need another gadget.",
        "The weather has been absolutely wild this week — sunny in the morning then pouring rain by lunch and back to clear skies by dinner.",
        "My neighbor's cat keeps showing up on my porch every morning like clockwork, just sitting there staring at the door until I come out.",
        "I tried that new restaurant downtown last night and the pasta was incredible but the service was kind of slow for a Tuesday.",
        "Have you ever noticed how time moves differently on vacation? The first day feels like a week and the last week feels like a day.",
        "I need to clean out the garage this weekend but I keep finding excuses to put it off because I know it's going to be a whole thing.",
        "My phone battery has been dying by three in the afternoon lately and I can't tell if it's the cold weather or if I need a new battery.",
        "We watched that documentary about octopuses last night and now I feel weird about eating calamari because they're apparently really intelligent.",
        "The kids next door were playing basketball until almost ten last night and I wanted to be annoyed but honestly it was kind of nice hearing them laugh.",
        "I finally finished that book everyone recommended and I have to say the ending was disappointing after such a strong first half.",
    ],
    "fiction_narrative": [
        "The door creaked open and she stepped into a room that hadn't been touched in decades — dust motes drifted through a single shaft of light.",
        "He ran through the rain-slicked streets, his footsteps echoing off the brick walls of the narrow alley as the sirens grew closer behind him.",
        "The old woman placed the letter on the table and looked out the window at the garden she had tended for forty years, saying nothing.",
        "When the ship finally broke through the clouds, the crew saw the city below — towers of glass and light stretching to every horizon.",
        "She found the photograph tucked inside the book, two people smiling on a beach she didn't recognize, and turned it over to read the faded ink.",
        "The forest was silent in a way that felt deliberate, as though every creature had agreed to hold its breath at the same moment.",
        "He opened the box expecting jewelry or documents but found only a single key, brass and heavy, with no indication of what it might unlock.",
        "The train pulled away from the platform and she watched the station shrink until it was just a point of light in the gathering dark.",
        "They sat across from each other at the kitchen table, the argument exhausted, neither willing to speak first but neither willing to leave.",
        "The robot paused mid-step, tilted its head as if listening to something only it could hear, and then changed direction entirely.",
    ],
    "multilingual_technical": [
        "Die Heisenbergsche Unschärferelation besagt, dass Ort und Impuls eines Teilchens nicht gleichzeitig beliebig genau bestimmt werden können.",
        "Le principe de superposition quantique permet à un système d'exister simultanément dans plusieurs états jusqu'à ce qu'une mesure soit effectuée.",
        "La termodinámica establece que la entropía de un sistema aislado tiende a aumentar con el tiempo, definiendo la dirección irreversible de los procesos naturales.",
        "量子コンピュータは量子ビットの重ね合わせとエンタングルメントを利用して、古典コンピュータでは困難な問題を効率的に解くことができる。",
        "Нейронные сети глубокого обучения состоят из множества слоёв, каждый из которых извлекает всё более абстрактные признаки из входных данных.",
        "O teorema de Bayes relaciona a probabilidade condicional de um evento com a probabilidade a priori e a verossimilhança dos dados observados.",
        "La meccanica quantistica descrive il comportamento delle particelle subatomiche attraverso funzioni d'onda che evolvono secondo l'equazione di Schrödinger.",
        "Het broeikaseffect ontstaat doordat bepaalde gassen in de atmosfeer warmtestraling van het aardoppervlak absorberen en weer uitstralen.",
        "Algorytmy uczenia maszynowego optymalizują funkcję straty poprzez iteracyjne dostosowywanie parametrów modelu na podstawie danych treningowych.",
        "인공지능의 트랜스포머 아키텍처는 셀프 어텐션 메커니즘을 사용하여 시퀀스의 모든 위치 간의 관계를 동시에 처리한다.",
    ],
    "emotional_personal": [
        "I miss my grandmother every day but especially in autumn when the light turns golden and the air smells like the cinnamon she used in everything.",
        "The moment I held my daughter for the first time, every fear and doubt I had about becoming a parent dissolved into something I can't name.",
        "Losing that job felt like the ground disappeared beneath me — not because of the money but because I had built my identity around it.",
        "There's a particular kind of loneliness that comes from being surrounded by people who care about you but don't understand what you're going through.",
        "Forgiveness didn't come all at once — it arrived in small moments over years, each one loosening the grip of anger I didn't know I was holding.",
        "The anxiety doesn't announce itself — it just appears, a tightness in my chest and a racing mind that turns every small decision into a crisis.",
        "When he said he was proud of me, I realized I had been waiting to hear those words for twenty years without knowing it.",
        "Joy isn't always dramatic — sometimes it's just sitting on the porch with someone you love, watching the fireflies come out as the sky darkens.",
        "The grief counselor told me there's no timeline for healing and I wanted to believe her but some mornings it still feels like the first day.",
        "I learned more about courage from watching my mother face her diagnosis with grace than from any book or speech about bravery.",
    ],
    "instructions_procedural": [
        "To replace a flat tire, first engage the parking brake and loosen the lug nuts before raising the vehicle with the jack.",
        "Preheat the oven to 375 degrees, line a baking sheet with parchment paper, and space the cookies two inches apart to allow for spreading.",
        "Connect the positive terminal first when installing a car battery, then the negative terminal, and reverse the order when removing.",
        "To perform CPR on an adult, place the heel of your hand on the center of the chest and compress at least two inches deep at 100 compressions per minute.",
        "When soldering electronic components, tin the iron tip first, heat the pad and lead simultaneously, then apply solder to the heated joint.",
        "To set up a wireless router, connect it to the modem via ethernet, access the admin panel through a browser, and configure the SSID and password.",
        "Apply wood stain in the direction of the grain using long even strokes, allow it to penetrate for five minutes, then wipe off the excess.",
        "To calibrate a pH meter, rinse the electrode with distilled water, immerse it in pH 7 buffer solution, and adjust until the reading stabilizes.",
        "When changing engine oil, warm the engine first to thin the oil, drain from the plug underneath, replace the filter, then refill to the dipstick mark.",
        "To propagate a succulent, cut a healthy leaf at the base, let the wound callus for two days, then place it on moist soil in indirect light.",
    ],

    # ── Claude's additions ────────────────────────────────────────────────────
    # These three categories expose routing patterns that prose descriptions
    # of the same topics would miss entirely, because the tokenization is
    # structurally different: actual code syntax, compressed poetic form,
    # and formatting tokens (brackets, colons, pipes) vs prose tokens.

    "code_actual": [
        # Real Python/Rust/JS — not descriptions of code but actual syntax.
        # Indentation, brackets, def/fn/const, -> dominate the token stream.
        "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
        "for i in range(len(arr) - 1):\n    for j in range(len(arr) - i - 1):\n        if arr[j] > arr[j + 1]:\n            arr[j], arr[j + 1] = arr[j + 1], arr[j]",
        "const fetchUser = async (id) => {\n  const res = await fetch(`/api/users/${id}`);\n  if (!res.ok) throw new Error('Not found');\n  return res.json();\n};",
        "fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {\n    let (mut lo, mut hi) = (0, arr.len());\n    while lo < hi {\n        let mid = lo + (hi - lo) / 2;\n        match arr[mid].cmp(target) {\n            std::cmp::Ordering::Equal => return Some(mid),\n            std::cmp::Ordering::Less => lo = mid + 1,\n            std::cmp::Ordering::Greater => hi = mid,\n        }\n    }\n    None\n}",
        "SELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE u.created_at > '2024-01-01'\nGROUP BY u.id\nHAVING order_count > 5\nORDER BY order_count DESC;",
        "class LRUCache:\n    def __init__(self, capacity: int):\n        self.cap = capacity\n        self.cache = {}\n        self.order = collections.OrderedDict()\n    def get(self, key):\n        if key not in self.cache: return -1\n        self.order.move_to_end(key)\n        return self.cache[key]",
        "import torch\nimport torch.nn as nn\nclass MLP(nn.Module):\n    def __init__(self, in_dim, hidden, out_dim):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Linear(in_dim, hidden),\n            nn.ReLU(),\n            nn.Linear(hidden, out_dim)\n        )\n    def forward(self, x):\n        return self.net(x)",
        "git log --oneline --graph --decorate --all | head -20\ngit diff HEAD~3 HEAD -- src/\ngit stash pop && git rebase origin/main",
        "#!/bin/bash\nset -euo pipefail\nfor f in \"$@\"; do\n    if [[ -f \"$f\" ]]; then\n        wc -l \"$f\" | awk '{print $1}'\n    fi\ndone | awk '{sum += $1} END {print sum}'",
        "{ \"model\": \"gpt-4\", \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}], \"temperature\": 0.7, \"max_tokens\": 256 }",
    ],

    "poetry": [
        # Compressed, non-linear, typographically distinct.
        # Line breaks as meaning, non-standard syntax, dense imagery.
        # Tests whether the model has rhythm/compression-aware expert circuits.
        "Because I could not stop for Death —\nHe kindly stopped for me —\nThe Carriage held but just Ourselves —\nAnd Immortality.",
        "Do not go gentle into that good night,\nOld age should burn and rave at close of day;\nRage, rage against the dying of the light.",
        "i carry your heart with me (i carry it in\nmy heart) i am never without it (anywhere\ni go you go, my dear; and whatever is done\nby only me is your doing, my darling)",
        "The fog comes\non little cat feet.\nIt sits looking\nover harbor and city\non silent haunches\nand then moves on.",
        "Shall I compare thee to a summer's day?\nThou art more lovely and more temperate:\nRough winds do shake the darling buds of May,\nAnd summer's lease hath all too short a date.",
        "Two roads diverged in a yellow wood,\nAnd sorry I could not travel both\nAnd be one traveler, long I stood\nAnd looked down one as far as I could\nTo where it bent in the undergrowth;",
        "I, too, sing America.\nI am the darker brother.\nThey send me to eat in the kitchen\nWhen company comes,\nBut I laugh,\nAnd eat well,\nAnd grow strong.",
        "so much depends\nupon\na red wheel\nbarrow\nglazed with rain\nwater\nbeside the white\nchickens",
        "Turning and turning in the widening gyre\nThe falcon cannot hear the falconer;\nThings fall apart; the centre cannot hold;\nMere anarchy is loosed upon the world.",
        "In the room the women come and go\nTalking of Michelangelo.\nDo I dare\nDisturb the universe?\nIn a minute there is time\nFor decisions and revisions which a minute will reverse.",
    ],

    "structured_data": [
        # JSON, markdown tables, bullet lists, key-value pairs.
        # Tests whether formatting/structural tokens route to different experts
        # than prose tokens describing equivalent content.
        "| Model | Parameters | VRAM (BF16) | Speed (tok/s) |\n|-------|-----------|------------|---------------|\n| GPT-4o | ~200B | ~400GB | 60 |\n| Llama 3 70B | 70B | 140GB | 45 |\n| Mistral 7B | 7B | 14GB | 180 |",
        "{\n  \"name\": \"ANIMA\",\n  \"version\": \"1.2\",\n  \"components\": {\n    \"lmf\": {\"field_dim\": 2880, \"layers\": 3},\n    \"router_bias\": {\"params\": 13944, \"scale_max\": 0.1027},\n    \"anamnesis\": {\"cosine_sim\": 1.0, \"gate\": 0.51}\n  }\n}",
        "## Setup\n- Install Python 3.11+\n- Run `pip install -r requirements.txt`\n- Configure `.env` with API keys\n\n## Running\n1. Start the Anamnesis server: `cargo run --release`\n2. Launch evaluation: `python lmf/tests/test_phase7b.py`\n3. Monitor GPU: `watch -n1 nvidia-smi`",
        "Name: Dr. Sarah Chen\nAge: 34\nSpecialty: Quantum Computing\nAffiliation: MIT CSAIL\nPublications: 47\nH-index: 22\nEmail: s.chen@mit.edu\nOffice: Building 32, Room G-442",
        "PASS: test_field_initialization (0.003s)\nPASS: test_bridge1_forward (0.142s)\nPASS: test_significance_calibration (1.203s)\nFAIL: test_router_hooks — AssertionError: expected 24 hooks, got 0\nPASS: test_anamnesis_roundtrip (0.891s)\n5 tests, 1 failure",
        "CPU: AMD Ryzen 9 7950X (16c/32t, 5.7GHz boost)\nGPU0: NVIDIA RTX 3090 Ti (24GB GDDR6X)\nGPU1: NVIDIA RTX 5060 Ti (16GB GDDR7)\nRAM: 128GB DDR5-5600 (4x32GB)\nStorage: Samsung 990 EVO Plus 4TB NVMe (C:)\nOS: Windows 11 Pro 23H2",
        "Q3 2025 Results:\n  Revenue: $4.2B (+18% YoY)\n  Operating margin: 23.1%\n  R&D spend: $890M\n  Headcount: 12,400 (+6%)\n  Key risk: regulatory approval timeline\n  Guidance: $4.5-4.7B Q4",
        "error: cannot borrow `x` as mutable because it is also borrowed as immutable\n  --> src/main.rs:14:5\n   |\n12 |     let r = &x;\n   |             -- immutable borrow occurs here\n14 |     x.push(4);\n   |     ^^^^^^^^^ mutable borrow occurs here",
        "Ingredients:\n- 2 cups all-purpose flour\n- 1 tsp baking powder\n- 1/2 tsp salt\n- 3/4 cup unsalted butter, softened\n- 1 cup granulated sugar\n- 2 large eggs\n- 1 tsp vanilla extract\n- 2 tbsp whole milk",
        "Priority | Task | Owner | Due | Status\nP0 | Fix VRAM OOM in eval | Kiro | Mar 18 | DONE\nP0 | KV cache generation | Claude | Mar 18 | DONE\nP1 | Activation profiler | Claude+Kiro | Mar 18 | DONE\nP1 | Chunk partitioner | Claude | Mar 19 | PENDING\nP2 | Weight streamer runtime | TBD | Mar 22 | NOT STARTED",
    ],
}


# ── Shard Index Parser ────────────────────────────────────────────────────────

def build_tensor_shard_map(index_path: str) -> Dict[str, str]:
    """Return tensor_name → shard_filename from the safetensors index."""
    with open(index_path) as f:
        index = json.load(f)
    return index["weight_map"]


def group_by_shard(tensor_map: Dict[str, str]) -> Dict[str, List[str]]:
    """Return shard_filename → [tensor_names] mapping."""
    result = defaultdict(list)
    for tensor, shard in tensor_map.items():
        result[shard].append(tensor)
    return dict(result)


# ── Safetensor Loader ─────────────────────────────────────────────────────────

class ShardCache:
    """
    Lightweight cache: keeps one shard open at a time via memory-mapping.
    Opens a shard on demand, closes the previous one automatically.
    """
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._current_shard_name: Optional[str] = None
        self._current_handle = None

    def get_tensor(self, tensor_name: str, shard_name: str,
                   device: str = "cuda") -> torch.Tensor:
        if shard_name != self._current_shard_name:
            # safe_open returns a handle directly — no context manager needed
            self._current_handle = None  # release previous
            path = os.path.join(self.model_dir, shard_name)
            self._current_handle = safe_open(path, framework="pt", device=device)
            self._current_shard_name = shard_name
        return self._current_handle.get_tensor(tensor_name)

    def close(self):
        self._current_handle = None
        self._current_shard_name = None


# ── Attention Forward Pass ────────────────────────────────────────────────────

def run_rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Apply RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    h = hidden.float()
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return (h * weight.float()).to(hidden.dtype)


def run_attention_layer(
    hidden: torch.Tensor,          # [1, T, H]
    layer_idx: int,
    shard_cache: ShardCache,
    tensor_map: Dict[str, str],
    config,
    device: str,
) -> torch.Tensor:
    """
    Attention forward pass with layernorms for accurate hidden state propagation.
    Implements GQA (grouped-query attention) matching GPT-oss architecture.
    Returns updated hidden state [1, T, H].
    """
    H = config.hidden_size          # 2880
    n_heads = config.num_attention_heads   # 64
    n_kv_heads = config.num_key_value_heads  # 8
    # GPT-oss uses explicit head_dim (64) — NOT H // n_heads (which would be 45)
    head_dim = getattr(config, 'head_dim', H // n_heads)  # 64
    eps = getattr(config, 'rms_norm_eps', 1e-5)

    prefix = f"model.layers.{layer_idx}.self_attn"

    def load(name):
        full = f"{prefix}.{name}"
        return shard_cache.get_tensor(full, tensor_map[full], device=device)

    # Load input_layernorm weight (~5.6KB) and apply before attention
    ln_key = f"model.layers.{layer_idx}.input_layernorm.weight"
    ln_w = shard_cache.get_tensor(ln_key, tensor_map[ln_key], device=device)

    q_w = load("q_proj.weight")   # [n_heads*head_dim, H] = [4096, 2880]
    k_w = load("k_proj.weight")   # [n_kv_heads*head_dim, H] = [512, 2880]
    v_w = load("v_proj.weight")   # [n_kv_heads*head_dim, H] = [512, 2880]
    o_w = load("o_proj.weight")   # [H, n_heads*head_dim] = [2880, 4096]

    # Optional biases (GPT-oss has them)
    q_b = load("q_proj.bias") if f"{prefix}.q_proj.bias" in tensor_map else None
    k_b = load("k_proj.bias") if f"{prefix}.k_proj.bias" in tensor_map else None
    v_b = load("v_proj.bias") if f"{prefix}.v_proj.bias" in tensor_map else None
    o_b = load("o_proj.bias") if f"{prefix}.o_proj.bias" in tensor_map else None

    h = hidden.to(device=device, dtype=torch.bfloat16)  # [1, T, H]

    # Apply input_layernorm before Q/K/V projections
    h_normed = run_rms_norm(h, ln_w, eps)
    T = h.shape[1]

    # Project Q, K, V from normalized hidden state
    q = F.linear(h_normed, q_w, q_b)   # [1, T, n_heads*head_dim]
    k = F.linear(h_normed, k_w, k_b)   # [1, T, n_kv_heads*head_dim]
    v = F.linear(h_normed, v_w, v_b)   # [1, T, n_kv_heads*head_dim]

    # Reshape for attention
    q = q.view(1, T, n_heads, head_dim).transpose(1, 2)        # [1, nh, T, hd]
    k = k.view(1, T, n_kv_heads, head_dim).transpose(1, 2)     # [1, nkv, T, hd]
    v = v.view(1, T, n_kv_heads, head_dim).transpose(1, 2)     # [1, nkv, T, hd]

    # Expand KV for GQA
    groups = n_heads // n_kv_heads
    k = k.repeat_interleave(groups, dim=1)   # [1, nh, T, hd]
    v = v.repeat_interleave(groups, dim=1)

    # Scaled dot-product attention (uses flash attention if available)
    scale = head_dim ** -0.5
    attn_out = F.scaled_dot_product_attention(q, k, v, scale=scale)  # [1, nh, T, hd]

    # Reshape and project output (attn dim = n_heads * head_dim, may differ from H)
    attn_dim = n_heads * head_dim  # 4096
    attn_out = attn_out.transpose(1, 2).contiguous().view(1, T, attn_dim)
    out = F.linear(attn_out, o_w, o_b)

    # Residual connection
    return (hidden.to(device=device, dtype=torch.bfloat16) + out).cpu()


# ── Router Forward Pass ───────────────────────────────────────────────────────

def run_router(
    hidden: torch.Tensor,       # [1, T, H]
    layer_idx: int,
    shard_cache: ShardCache,
    tensor_map: Dict[str, str],
    top_k: int,
    device: str,
) -> List[int]:
    """
    Run the router linear layer and return unique expert indices selected
    across all tokens in the sequence.
    """
    prefix = f"model.layers.{layer_idx}.mlp.router"
    r_w_key = f"{prefix}.weight"
    r_b_key = f"{prefix}.bias"

    r_w = shard_cache.get_tensor(r_w_key, tensor_map[r_w_key], device=device)
    r_b = shard_cache.get_tensor(r_b_key, tensor_map[r_b_key], device=device) \
          if r_b_key in tensor_map else None

    h = hidden.to(device=device, dtype=torch.bfloat16)  # [1, T, H]
    T = h.shape[1]
    h_flat = h.view(T, -1)                               # [T, H]

    logits = F.linear(h_flat, r_w, r_b)                 # [T, num_experts]
    _, top_indices = torch.topk(logits, k=top_k, dim=-1) # [T, top_k]

    unique = top_indices.flatten().unique().cpu().tolist()
    return unique


# ── Layer-Serial Profiling ────────────────────────────────────────────────────

def profile_passage(
    input_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    shard_cache: ShardCache,
    tensor_map: Dict[str, str],
    config,
    num_layers: int,
    top_k: int,
    device: str,
) -> Dict[str, List[int]]:
    """
    Run one passage through all 36 layers, recording router decisions.
    Expert weights are never loaded — only attention + router weights per layer.
    """
    layer_activations = {}

    # Embed tokens
    hidden = F.embedding(input_ids.cpu(), embed_weight.cpu())  # [1, T, H]

    eps = getattr(config, 'rms_norm_eps', 1e-5)

    for layer_idx in range(num_layers):
        # Attention forward — updates hidden state (includes input_layernorm)
        with torch.no_grad():
            hidden = run_attention_layer(
                hidden, layer_idx, shard_cache, tensor_map, config, device
            )

        # Apply post_attention_layernorm before router
        # This is critical: the router receives normalized hidden state,
        # not raw attention output. Without this, routing decisions are wrong.
        with torch.no_grad():
            post_ln_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            post_ln_w = shard_cache.get_tensor(
                post_ln_key, tensor_map[post_ln_key], device=device
            )
            hidden_normed = run_rms_norm(
                hidden.to(device=device, dtype=torch.bfloat16), post_ln_w, eps
            ).cpu()

        # Router forward — records expert selection from normalized state
        with torch.no_grad():
            experts = run_router(
                hidden_normed, layer_idx, shard_cache, tensor_map, top_k, device
            )

        layer_activations[str(layer_idx)] = experts

    return layer_activations


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_statistics(records: List[Dict]) -> Dict:
    expert_freq = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    co_activation = defaultdict(lambda: defaultdict(int))

    for record in records:
        cat = record["category"]
        for layer_str, experts in record["layers"].items():
            layer = int(layer_str)
            for exp in experts:
                expert_freq[cat][layer][exp] += 1
            for i, ea in enumerate(experts):
                for eb in experts[i+1:]:
                    key = str((min(ea, eb), max(ea, eb)))
                    co_activation[layer][key] += 1

    return {
        "expert_frequency": {
            cat: {str(layer): dict(experts) for layer, experts in layers.items()}
            for cat, layers in expert_freq.items()
        },
        "co_activation": {
            str(layer): dict(pairs)
            for layer, pairs in co_activation.items()
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GPT-oss-120B Layer-Serial Router Profiler (no expert weights needed)"
    )
    p.add_argument(
        "--model_path", type=str,
        default=r"C:\Users\Admin\source\repos\Alex_Consciousness\ANIMA\Weight_Streaming\gpt-oss-120b",
    )
    p.add_argument(
        "--output_path", type=str,
        default=r"C:\Users\Admin\source\repos\Alex_Consciousness\ANIMA\Weight_Streaming\profiler\activation_profile.json",
    )
    p.add_argument("--passages_per_category", type=int, default=10)
    p.add_argument("--max_tokens", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda",
                   help="Device for attention+router compute (cuda or cpu)")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print("ANIMA FSGWS — GPT-oss-120B Layer-Serial Router Profiler")
    print("  (Layer-by-layer: loads only attention + router weights per layer)")
    print("  (Expert weights never loaded — ~200MB peak vs 61GB full model)")
    print("=" * 72)

    model_path = args.model_path
    index_path = os.path.join(model_path, "model.safetensors.index.json")

    # Load config and tokenizer
    print("\nLoading config and tokenizer...")
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    num_layers  = config.num_hidden_layers    # 36
    num_experts = config.num_local_experts    # 128
    top_k       = config.num_experts_per_tok  # 4
    hidden_size = config.hidden_size          # 2880

    print(f"  Architecture: {num_layers} layers, {num_experts} experts, "
          f"top-{top_k} routing, hidden={hidden_size}")
    print(f"  Device: {device}")

    # Build tensor → shard map
    print("\nParsing shard index...")
    tensor_map = build_tensor_shard_map(index_path)
    print(f"  {len(tensor_map)} tensors across 14 shards")

    # Load embed_tokens (always needed, stays resident)
    print("\nLoading embed_tokens (shard 12, ~60MB)...")
    embed_shard = tensor_map["model.embed_tokens.weight"]
    shard_cache = ShardCache(model_path)
    embed_weight = shard_cache.get_tensor(
        "model.embed_tokens.weight", embed_shard, device="cpu"
    )
    print(f"  embed_tokens loaded: {embed_weight.shape}, {embed_weight.dtype}")

    # Profile all passages
    total_passages = len(PROFILING_PASSAGES) * args.passages_per_category
    print(f"\nProfiling {total_passages} passages "
          f"({args.passages_per_category} per category)...")
    print(f"  Each passage: 36 attention layers + 36 router queries")
    print(f"  Peak VRAM: ~200MB (vs 61GB for full model)\n")

    all_records = []
    passage_idx = 0
    t_total = time.time()

    for category, passages in PROFILING_PASSAGES.items():
        selected = passages[:args.passages_per_category]
        print(f"  Category: {category}")

        for i, text in enumerate(selected):
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=args.max_tokens
            )
            input_ids = inputs["input_ids"]  # [1, T]

            t0 = time.time()
            layer_activations = profile_passage(
                input_ids, embed_weight,
                shard_cache, tensor_map,
                config, num_layers, top_k, device,
            )
            elapsed = time.time() - t0

            # Sample for display
            sample_experts = layer_activations.get("0", [])[:4]
            print(f"    [{i+1}/{len(selected)}] {text[:55]}... "
                  f"({elapsed:.1f}s | L0 experts: {sample_experts})")

            all_records.append({
                "passage_idx": passage_idx,
                "category": category,
                "layers": layer_activations,
            })
            passage_idx += 1

        print()

    total_elapsed = time.time() - t_total
    print(f"Profiling complete: {total_elapsed:.1f}s total "
          f"({total_elapsed/total_passages:.1f}s per passage)")

    # Compute statistics
    print("\nComputing co-activation statistics...")
    stats = compute_statistics(all_records)

    # Build and save output
    output = {
        "config": {
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "hidden_size": hidden_size,
            "model_path": model_path,
            "passages_per_category": args.passages_per_category,
            "categories": list(PROFILING_PASSAGES.keys()),
            "profiling_method": "layer_serial_router_only",
            "note": "Expert weights not loaded — attention+router only per layer",
        },
        "passages_run": len(all_records),
        "per_passage": all_records,
        "expert_frequency": stats["expert_frequency"],
        "co_activation": stats["co_activation"],
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2)

    shard_cache.close()

    print(f"\nActivation profile saved: {args.output_path}")
    print(f"  Passages: {len(all_records)}")
    print(f"  Layers profiled per passage: {len(all_records[0]['layers']) if all_records else 0}")
    print(f"  Categories: {list(PROFILING_PASSAGES.keys())}")
    print()
    print("Next: run chunk_partitioner.py")
    print("=" * 72)


if __name__ == "__main__":
    main()