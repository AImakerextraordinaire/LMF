"""
ANIMA Phase 3 - Expanded Training Dataset
Diverse passages with clear context->target reference structure.
Covers: science, history, geography, technology, biology, arts, engineering.
"""

TRAINING_PASSAGES_EXPANDED = [
    # === ORIGINAL 8 (keep for continuity) ===
    {
        "context": "The ancient city of Petra was carved into rose-red cliffs by the Nabataean people over two thousand years ago. Located in modern-day Jordan, Petra served as a crucial trading hub connecting Arabia, Egypt, and the Mediterranean. The Nabataeans were master hydraulic engineers who built an elaborate system of dams, cisterns, and water channels to sustain their desert city.",
        "target": "Today, Petra remains one of the most remarkable archaeological sites in the world. The Treasury, carved directly into the sandstone cliff face, is perhaps the most iconic structure from this ancient Nabataean civilization. Visitors can still see traces of the sophisticated water management systems that once sustained life in this desert trading center."
    },
    {
        "context": "Marie Curie was born Maria Sklodowska in Warsaw, Poland in 1867. She moved to Paris to study physics and mathematics at the Sorbonne, where she met Pierre Curie. Together, they discovered two new elements: polonium, named after Marie's homeland of Poland, and radium. Marie became the first woman to win a Nobel Prize.",
        "target": "Her groundbreaking research on radioactivity earned her not one but two Nobel Prizes, making her the first person to win Nobel Prizes in two different sciences. The element polonium, which she named in honor of her native Poland, was discovered through her painstaking work isolating radioactive compounds. Marie Curie's legacy in physics and chemistry continues to inspire scientists worldwide."
    },
    {
        "context": "The octopus is one of the most intelligent invertebrates on Earth. It has three hearts, blue blood, and eight arms lined with suckers. Each arm contains a cluster of neurons, giving octopuses a distributed nervous system. They can change color and texture in milliseconds to camouflage themselves, and have been observed using tools like coconut shells for shelter.",
        "target": "Scientists studying octopus cognition have been amazed by their problem-solving abilities. Their distributed nervous system, with neural clusters in each of their eight arms, allows for remarkably complex behavior. These creatures with three hearts and blue blood can open jars, navigate mazes, and even recognize individual human faces, demonstrating intelligence that rivals many vertebrates."
    },
    {
        "context": "The Great Barrier Reef stretches over 2,300 kilometers along the northeast coast of Australia. It is the largest living structure on Earth, visible from space. The reef is home to over 1,500 species of fish, 400 types of coral, and countless other marine organisms. Rising ocean temperatures pose the greatest threat to this ecosystem through coral bleaching.",
        "target": "Conservation efforts for the Great Barrier Reef have intensified as ocean warming accelerates coral bleaching events along Australia's northeastern coastline. The reef, which spans more than two thousand kilometers and supports extraordinary biodiversity including thousands of fish species, faces an uncertain future. Protecting this massive living structure, the largest on Earth, requires global action on climate change."
    },
    {
        "context": "Johann Sebastian Bach composed over a thousand works during his lifetime, including the Brandenburg Concertos, the Mass in B minor, and The Well-Tempered Clavier. Working in Germany during the Baroque period, Bach served as a church musician and court composer. His mastery of counterpoint and harmony laid the foundation for Western classical music theory.",
        "target": "The influence of Bach on Western music cannot be overstated. His works, from the Brandenburg Concertos to The Well-Tempered Clavier, established fundamental principles of harmony and counterpoint that composers still study today. This Baroque-era church musician and court composer from Germany created a body of over a thousand compositions that represents one of the greatest achievements in the history of music."
    },
    {
        "context": "The human brain contains approximately 86 billion neurons, each connected to thousands of others through synapses. The prefrontal cortex handles executive function and decision-making, while the hippocampus is crucial for memory formation and spatial navigation. During sleep, the brain consolidates memories by replaying experiences and transferring them from short-term to long-term storage.",
        "target": "Memory consolidation during sleep is one of the most fascinating processes in neuroscience. The hippocampus replays recently formed memories while we sleep, gradually transferring important experiences to long-term storage across the cortex. This process, occurring in a brain with billions of neurons and trillions of synaptic connections, helps explain why adequate sleep is essential for learning and memory."
    },
    {
        "context": "Rust is a systems programming language focused on safety, speed, and concurrency. Created by Graydon Hoare at Mozilla, Rust uses an ownership system with borrowing rules enforced at compile time to prevent data races and memory leaks. Unlike C and C++, Rust achieves memory safety without garbage collection.",
        "target": "The ownership model that defines Rust has made it increasingly popular for systems where reliability is critical. By enforcing borrowing rules at compile time rather than relying on garbage collection, Rust eliminates entire categories of bugs including data races and memory leaks. This systems language, originally developed at Mozilla, proves that safety and performance need not be opposing goals."
    },
    {
        "context": "Mount Everest stands at 8,849 meters above sea level, making it the highest point on Earth. Located in the Himalayas on the border of Nepal and Tibet, it was first summited by Edmund Hillary and Tenzing Norgay in 1953. The mountain is known by several names: Chomolungma in Tibetan and Sagarmatha in Nepali.",
        "target": "Since Hillary and Norgay first reached its summit in 1953, thousands of climbers have attempted to scale this Himalayan peak. Known as Sagarmatha in Nepal and Chomolungma in Tibet, the mountain that straddles their border rises nearly nine thousand meters above sea level. Mount Everest, the highest point on Earth, continues to draw adventurers from around the world."
    },

    # === NEW PASSAGES (24 more for diverse coverage) ===
    
    # --- Physics / Space ---
    {
        "context": "Black holes are regions of spacetime where gravity is so intense that nothing, not even light, can escape. They form when massive stars collapse at the end of their life cycle. The boundary around a black hole is called the event horizon. In 2019, the Event Horizon Telescope captured the first direct image of a black hole in the galaxy M87, showing a bright ring of superheated gas surrounding a dark shadow.",
        "target": "The historic first image of a black hole captured by the Event Horizon Telescope revealed the shadow and accretion disk of the supermassive object at the center of galaxy M87. This observation confirmed decades of theoretical predictions about event horizons and the behavior of light near extreme gravitational fields. The bright ring visible in the image is superheated gas orbiting at nearly the speed of light before crossing the point of no return."
    },
    {
        "context": "The Voyager 1 spacecraft, launched by NASA in 1977, is the most distant human-made object in space. It carries a Golden Record containing sounds and images of Earth, intended as a message for any extraterrestrial civilization that might find it. Voyager 1 entered interstellar space in 2012, becoming the first spacecraft to leave the heliosphere, the bubble of solar wind surrounding our solar system.",
        "target": "More than four decades after its launch, Voyager 1 continues to transmit data from beyond the heliosphere, having crossed into interstellar space. This NASA spacecraft, carrying its famous Golden Record with sounds and images representing Earth, remains our most distant emissary to the cosmos. The data it sends back about conditions outside the protective bubble of solar wind has transformed our understanding of the boundary between our solar system and the galaxy beyond."
    },
    
    # --- Biology ---
    {
        "context": "CRISPR-Cas9 is a revolutionary gene-editing technology derived from a natural defense mechanism found in bacteria. Bacteria use CRISPR sequences to recognize and cut the DNA of invading viruses. Jennifer Doudna and Emmanuelle Charpentier adapted this system into a precise molecular tool that can edit any DNA sequence in living organisms, earning them the 2020 Nobel Prize in Chemistry.",
        "target": "The Nobel Prize awarded to Doudna and Charpentier recognized their development of CRISPR-Cas9 into a programmable gene-editing tool. Originally discovered as a bacterial immune defense against viruses, this molecular system can now precisely cut and modify DNA sequences in plants, animals, and humans. The implications for treating genetic diseases, developing crops, and understanding biology are profound and still unfolding."
    },
    {
        "context": "Tardigrades, also known as water bears, are microscopic animals renowned for their extraordinary survival abilities. They can withstand temperatures from near absolute zero to above 150 degrees Celsius, survive radiation doses hundreds of times lethal to humans, and endure the vacuum of outer space. When conditions become hostile, tardigrades enter a state called cryptobiosis, essentially suspending all metabolic processes.",
        "target": "The ability of tardigrades to survive extreme conditions has fascinated scientists for decades. These microscopic water bears achieve near-indestructibility through cryptobiosis, a state where metabolic activity essentially stops. In this suspended state, they have survived exposure to outer space, extreme radiation, and temperature ranges that would instantly kill virtually any other organism. Understanding how they protect their DNA and cellular structures during cryptobiosis could lead to breakthroughs in medicine and space travel."
    },

    # --- History ---
    {
        "context": "The Library of Alexandria, founded in the 3rd century BCE under Ptolemy I, was the largest and most significant library of the ancient world. It aimed to collect all the knowledge of the known world, housing an estimated 400,000 to 700,000 scrolls. Scholars from across the Mediterranean came to study there, making Alexandria the intellectual capital of the ancient world. The library's destruction, which occurred gradually over several centuries, is considered one of the greatest losses of knowledge in human history.",
        "target": "The loss of the Library of Alexandria remains one of history's most lamented intellectual catastrophes. Founded by Ptolemy and housing hundreds of thousands of scrolls, this institution in ancient Alexandria attracted the greatest minds of the Mediterranean world. The knowledge contained in those scrolls, representing centuries of scholarship and discovery, was irreplaceable. The gradual destruction of this great library serves as a powerful reminder of the fragility of human knowledge."
    },
    {
        "context": "The Rosetta Stone, discovered in 1799 by French soldiers during Napoleon's Egyptian campaign, contained the same text written in three scripts: hieroglyphics, Demotic, and ancient Greek. Jean-Francois Champollion used the Greek text as a key to decipher Egyptian hieroglyphics in 1822, unlocking a writing system that had been unreadable for nearly two thousand years.",
        "target": "Champollion's decipherment of hieroglyphics using the Rosetta Stone was one of the great intellectual achievements of the 19th century. By comparing the Greek inscription with the hieroglyphic text on the stone discovered during Napoleon's Egyptian expedition, he cracked a code that had baffled scholars for centuries. This breakthrough opened up the entire history of ancient Egypt, allowing researchers to read inscriptions on temples, tombs, and papyri for the first time in nearly two millennia."
    },

    # --- Technology ---
    {
        "context": "The Apollo Guidance Computer, designed at MIT, was one of the first computers to use integrated circuits. It had approximately 74 kilobytes of memory and operated at 0.043 MHz. Despite being less powerful than a modern calculator, this computer successfully navigated Apollo 11 to the Moon and back in 1969. Margaret Hamilton, who led the software engineering team, coined the term 'software engineering' during the project.",
        "target": "Margaret Hamilton's software for the Apollo Guidance Computer proved its worth during the actual Moon landing when unexpected alarms threatened to abort the mission. Her team's robust error-handling design allowed the computer, with its tiny 74 kilobytes of memory, to prioritize essential navigation tasks. The fact that astronauts reached the Moon using a machine less powerful than a modern calculator stands as a testament to brilliant engineering, and Hamilton's contributions helped establish software engineering as a recognized discipline."
    },
    {
        "context": "Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN, the European particle physics laboratory in Geneva. He created the first web browser, the first web server, and the first website. His original proposal described a system of interlinked hypertext documents accessible via the internet. Berners-Lee made the technology freely available, refusing to patent it, which allowed the web to grow into a global phenomenon.",
        "target": "The decision by Berners-Lee to make the World Wide Web freely available rather than patenting it was perhaps the most consequential act of technological generosity in modern history. His invention at CERN, built on the concept of interlinked hypertext documents, transformed from an academic tool into the foundation of modern digital life. The first website, the first browser, and the first server all originated from his work in Geneva, and his choice to keep the technology open ensured that the web belonged to everyone."
    },

    # --- Geography / Nature ---
    {
        "context": "Iceland sits on the Mid-Atlantic Ridge, where the North American and Eurasian tectonic plates are slowly pulling apart. This creates intense geothermal activity, with hot springs, geysers, and volcanoes scattered across the island. The country generates nearly 100 percent of its electricity from renewable sources, primarily geothermal and hydroelectric power. The original geyser, from which all geysers take their name, is Geysir in southwestern Iceland.",
        "target": "The remarkable renewable energy achievement of Iceland stems directly from its position on the Mid-Atlantic Ridge. The tectonic forces pulling the North American and Eurasian plates apart create the geothermal activity that powers the nation. Hot springs and geysers, including the original Geysir that gave all geysers their name, are surface expressions of the same volcanic energy that generates nearly all of the country's electricity without fossil fuels."
    },
    {
        "context": "The Mariana Trench in the western Pacific Ocean contains the deepest point on Earth's surface, known as the Challenger Deep, which reaches approximately 10,994 meters below sea level. In 1960, Jacques Piccard and Don Walsh became the first humans to reach the bottom in the bathyscaphe Trieste. The pressure at the bottom exceeds 1,000 atmospheres, yet life still exists there, including amphipods, sea cucumbers, and xenophyophores.",
        "target": "Life at the bottom of the Mariana Trench defies expectations about the limits of biology. At the Challenger Deep, nearly eleven kilometers below the surface, organisms withstand pressures exceeding a thousand atmospheres. Since Piccard and Walsh first descended in the Trieste in 1960, subsequent expeditions have discovered thriving communities of amphipods and other creatures in this extreme environment, demonstrating that life can adapt to conditions far beyond what was once thought possible."
    },

    # --- Mathematics / Computer Science ---
    {
        "context": "Alan Turing published his seminal paper 'On Computable Numbers' in 1936, introducing the concept of the Turing machine, a theoretical device that could simulate any algorithm. During World War II, Turing worked at Bletchley Park where he designed the Bombe machine to crack the German Enigma cipher. His 1950 paper 'Computing Machinery and Intelligence' proposed the Turing test as a measure of machine intelligence.",
        "target": "The intellectual legacy of Turing spans from the theoretical foundations of computer science to practical cryptanalysis to artificial intelligence. His Turing machine concept from 1936 established the mathematical limits of computation. At Bletchley Park, his Enigma-breaking work with the Bombe machine shortened the war significantly. And his famous test for machine intelligence, proposed in 1950, continues to frame debates about AI to this day."
    },
    {
        "context": "The Fibonacci sequence begins with 0 and 1, with each subsequent number being the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, 21, and so on. The ratio between consecutive Fibonacci numbers converges to the golden ratio, approximately 1.618. This mathematical pattern appears throughout nature in the arrangement of leaves, the spirals of shells, the branching of trees, and the pattern of seeds in sunflower heads.",
        "target": "The prevalence of the Fibonacci sequence in natural structures suggests a deep connection between mathematics and biology. Sunflower seed spirals, nautilus shell curves, and the branching patterns of trees all follow ratios that converge on the golden ratio of approximately 1.618. This number, emerging from the simple rule of adding consecutive terms starting from 0 and 1, appears to be a fundamental organizing principle in how living systems grow and develop."
    },

    # --- Engineering ---
    {
        "context": "The Panama Canal, completed in 1914 after a decade of construction, connects the Atlantic and Pacific Oceans through a system of locks that raise and lower ships 26 meters above sea level. The canal uses Gatun Lake, an artificial lake created by damming the Chagres River, as its central waterway. Approximately 14,000 ships pass through the canal annually, avoiding the 12,000 kilometer journey around Cape Horn at the southern tip of South America.",
        "target": "The engineering achievement of the Panama Canal transformed global shipping by eliminating the need for the dangerous twelve-thousand-kilometer voyage around Cape Horn. The system of locks that raises vessels twenty-six meters to the level of Gatun Lake, the artificial reservoir at the heart of the canal, was a masterpiece of early twentieth-century engineering. Today, thousands of ships annually transit this waterway connecting the Atlantic and Pacific, making it one of the most important commercial passages in the world."
    },
    {
        "context": "The International Space Station orbits Earth at approximately 400 kilometers altitude, traveling at 28,000 kilometers per hour and completing one orbit every 90 minutes. It has been continuously inhabited since November 2000, making it the longest continuous human presence in space. The station is a collaborative project involving NASA, Roscosmos, ESA, JAXA, and CSA, with modules contributed by different countries assembled over more than a decade of spacewalks and robotic operations.",
        "target": "The continuous human habitation of the International Space Station for over two decades represents an unprecedented achievement in international cooperation and space engineering. Orbiting at four hundred kilometers altitude and traveling at twenty-eight thousand kilometers per hour, the station circles the Earth every ninety minutes. Built through collaboration between five space agencies and assembled through years of complex spacewalks, it serves as both a laboratory and a symbol of what nations can accomplish together in the pursuit of scientific knowledge."
    },

    # --- Arts / Culture ---
    {
        "context": "The Sistine Chapel ceiling, painted by Michelangelo between 1508 and 1512, contains over 300 figures across more than 500 square meters. Pope Julius II commissioned the work, which depicts scenes from the Book of Genesis, including the iconic Creation of Adam. Michelangelo painted much of the ceiling while standing on scaffolding with his arms raised overhead, a physically grueling process that took four years to complete.",
        "target": "Michelangelo's four years of labor on the Sistine Chapel ceiling produced one of humanity's greatest artistic achievements. Working on scaffolding with arms raised for hours at a time, he covered more than five hundred square meters with over three hundred figures. The commission from Pope Julius II to depict Genesis scenes resulted in images that defined Western art, none more famous than the Creation of Adam, where the outstretched fingers of God and man nearly touch."
    },
    {
        "context": "Shakespeare wrote approximately 37 plays and 154 sonnets during his career in Elizabethan and Jacobean England. His works introduced an estimated 1,700 words to the English language, including 'assassination,' 'lonely,' and 'eyeball.' The Globe Theatre, where many of his plays were performed, was a wooden open-air playhouse on the south bank of the Thames in London. Shakespeare's influence on literature and the English language remains unparalleled.",
        "target": "The linguistic legacy of Shakespeare extends far beyond his plays and sonnets. The approximately 1,700 words he contributed to English, from 'assassination' to 'lonely,' are still in everyday use four centuries later. His works, performed at the Globe Theatre on the Thames and across London's stages, explored the full range of human experience with a depth that continues to resonate. No single writer has shaped the English language more profoundly than this playwright from Elizabethan England."
    },

    # --- Medicine ---
    {
        "context": "Alexander Fleming discovered penicillin in 1928 when he noticed that a mold called Penicillium notatum had contaminated one of his petri dishes and killed the surrounding bacteria. Howard Florey and Ernst Boris Chain later developed methods to mass-produce the drug during World War II. Penicillin became the first widely used antibiotic, saving an estimated 200 million lives since its introduction and earning Fleming, Florey, and Chain the 1945 Nobel Prize in Physiology or Medicine.",
        "target": "The accidental discovery of penicillin by Fleming transformed medicine forever. The observation that Penicillium mold killed bacteria in a contaminated petri dish led to the development of the first mass-produced antibiotic, thanks to the work of Florey and Chain. Since its introduction during World War II, penicillin has saved hundreds of millions of lives. The 1945 Nobel Prize recognized all three scientists for launching the antibiotic era that fundamentally changed how humanity fights infectious disease."
    },
    {
        "context": "The Human Genome Project, launched in 1990 and completed in 2003, identified and mapped all approximately 20,500 genes in human DNA. The project involved researchers from 20 institutions across 6 countries and cost approximately 2.7 billion dollars. The entire human genome consists of about 3.2 billion base pairs of DNA. The project's completion has enabled personalized medicine, genetic testing, and a deeper understanding of hereditary diseases.",
        "target": "The completion of the Human Genome Project in 2003 opened a new era in medicine and biology. Mapping all twenty thousand genes and three billion base pairs of human DNA required over a decade of international collaboration and billions of dollars in funding. The knowledge gained has transformed genetic research, enabling doctors to identify hereditary disease risks, develop targeted therapies, and move toward truly personalized medicine based on individual genetic profiles."
    },

    # --- Philosophy / Social Science ---
    {
        "context": "The printing press, invented by Johannes Gutenberg in Mainz, Germany around 1440, used movable metal type to mechanize the production of books. His first major printed work was the Gutenberg Bible, completed around 1455. Before the printing press, books were copied by hand, making them rare and expensive. Within fifty years of Gutenberg's invention, an estimated twenty million books had been printed across Europe, dramatically increasing literacy and the spread of ideas.",
        "target": "Gutenberg's printing press triggered an information revolution comparable to the internet. The introduction of movable metal type in Mainz allowed books to be produced at a scale previously unimaginable. Where hand-copied manuscripts had been luxury items accessible only to the wealthy, printed books became available to a growing middle class. The millions of volumes produced in the decades following the Gutenberg Bible's publication in 1455 fundamentally transformed European society by democratizing access to knowledge."
    },

    # --- Environmental Science ---
    {
        "context": "The ozone layer, located in the stratosphere between 15 and 35 kilometers above Earth's surface, absorbs most of the Sun's harmful ultraviolet radiation. In the 1980s, scientists discovered a dramatic thinning of this layer over Antarctica, caused by chlorofluorocarbons (CFCs) used in refrigerants and aerosol sprays. The Montreal Protocol, signed in 1987, banned the production of CFCs. NASA has confirmed that the ozone hole has been slowly shrinking since the early 2000s.",
        "target": "The recovery of the ozone layer stands as one of the greatest environmental success stories. After scientists identified CFCs as the cause of the Antarctic ozone hole, the Montreal Protocol's worldwide ban on these chemicals demonstrated that international cooperation could address global environmental threats. NASA observations confirming the gradual shrinking of the ozone hole since 2000 validate the effectiveness of this approach. The stratospheric ozone layer, which shields life from harmful ultraviolet radiation, is slowly healing because nations chose to act on scientific evidence."
    },

    # --- Materials Science ---
    {
        "context": "Graphene is a single layer of carbon atoms arranged in a hexagonal lattice, making it the thinnest material known to science at just one atom thick. Despite this, it is approximately 200 times stronger than steel and conducts electricity better than copper. Andre Geim and Konstantin Novoselov first isolated graphene in 2004 using adhesive tape to peel layers from graphite, earning them the 2010 Nobel Prize in Physics.",
        "target": "The remarkable properties of graphene have generated enormous research interest since Geim and Novoselov isolated it using their famous scotch tape method. This one-atom-thick sheet of carbon arranged in hexagonal patterns combines extraordinary strength, superior electrical conductivity, and near-perfect transparency. The 2010 Nobel Prize recognized the profound implications of their discovery, and researchers continue to explore applications ranging from flexible electronics to water filtration to next-generation batteries."
    },

    # --- Astronomy ---
    {
        "context": "The James Webb Space Telescope, launched on December 25, 2021, is the most powerful space telescope ever built. Its primary mirror spans 6.5 meters across, composed of 18 hexagonal gold-coated beryllium segments. JWST orbits the Sun at the L2 Lagrange point, approximately 1.5 million kilometers from Earth. Operating primarily in infrared wavelengths, it can detect light from the earliest galaxies formed after the Big Bang, over 13 billion years ago.",
        "target": "The infrared observations from the James Webb Space Telescope have already transformed our understanding of the early universe. Stationed at the L2 Lagrange point, its 6.5-meter gold-coated mirror captures light that has traveled over thirteen billion years from the first galaxies. Since its Christmas Day 2021 launch, JWST has revealed structures in the early cosmos that challenge existing models of galaxy formation, proving that our most powerful telescope is reshaping astronomy with every new observation."
    },

    # --- Neuroscience / AI ---
    {
        "context": "Neural networks, inspired by the structure of biological brains, use layers of interconnected nodes to process information. Geoffrey Hinton, Yann LeCun, and Yoshua Bengio are often called the 'godfathers of deep learning' for their pioneering work on backpropagation and convolutional neural networks. The transformer architecture, introduced by Vaswani et al. in the 2017 paper 'Attention Is All You Need,' replaced recurrent networks with self-attention mechanisms and enabled the creation of large language models.",
        "target": "The evolution from early neural networks to modern large language models represents decades of accumulated insight. The foundational work by Hinton, LeCun, and Bengio on backpropagation and convolutional architectures established deep learning as a viable approach. The breakthrough came with the transformer architecture and its self-attention mechanism, described in the landmark 'Attention Is All You Need' paper. This shift from recurrence to attention enabled models to process sequences in parallel, unlocking the scale that makes today's AI systems possible."
    },
]
