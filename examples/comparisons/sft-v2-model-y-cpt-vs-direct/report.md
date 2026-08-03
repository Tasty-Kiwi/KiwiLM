# KiwiLM SFT instruction-adherence report

Scores use deterministic lexical checks. Lower repetition is better.

## Aggregate scores

| Model | Profile | Words | Summary | Features | Entities | Adherence | Repeat-4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model-Y-Direct-SFT-v2 | greedy | 55.6% | 45.8% | 83.3% | 50.0% | 59.6% | 15.0% |
| Model-Y-CPT-SFT-v2 | greedy | 50.0% | 41.7% | 66.7% | 50.0% | 52.7% | 12.0% |
| Model-Y-Direct-SFT-v2 | focused | 55.6% | 50.0% | 100.0% | 50.0% | 65.2% | 5.0% |
| Model-Y-CPT-SFT-v2 | focused | 66.7% | 54.2% | 100.0% | 50.0% | 69.0% | 3.5% |

## friends_before_dark / greedy / Model-Y-Direct-SFT-v2

- Words: 66.7% (oak, kind)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 8.7%

```text
Once upon a time, there was a kind girl named Lily. She loved to play outside in the sun. One day, she saw a big oak tree. She wanted to climb it, but it was too high. 

Lily asked her friend, a little bird, to help her. The bird said, "I can help you climb the oak tree." Lily was happy and said, "Thank you, bird!" 

As they climbed the oak tree, they saw a big, scary tree. Lily said, "I want to climb the tree!" The bird said, "No, we can't climb the oak tree." 

Lily and the bird started to climb the oak tree. They were having so much fun! But then, they saw a big, scary tree. The tree was very scary and scary. Lily and the bird were scared and ran away. 

But then, they saw a big, scary tree. They were scared and didn't know what to do
```

## friends_before_dark / greedy / Model-Y-CPT-SFT-v2

- Words: 66.7% (oak, kind)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 10.2%

```text
Once upon a time, there was a kind girl named Lily. She loved to play outside in the sun. One day, she saw a big oak tree. She wanted to climb it, but she was scared.

Lily asked her friend, "Can I climb the tree?" Her friend said, "Yes, you can climb the tree." They climbed up the tree and climbed up the tree. They were so happy!

After they reached the top, they saw a big tree. Lily said, "Look, I found a big tree!" Her friend said, "Yes, it is. Let's climb it together." They climbed up the tree and climbed up the tree. They felt happy and safe.
```

## friends_before_dark / focused / Model-Y-Direct-SFT-v2

- Words: 66.7% (oak, kind)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 3.4%

```text
Once upon a time, there was a kind girl named Lily. She loved to play outside in the sun. One day, she found a small oak tree in the woods. She wanted to climb it, but it was too high up.

Lily asked her friend, a little girl named Lily, to help her. Lily said, "Let's climb the oak tree together!" They climbed up the tree and climbed the oak tree. They were having so much fun!

After they finished the oak tree, they went back home. Lily's mom was not sad anymore. She said, "Lily, you should always be kind to others. We should always be kind to others." Lily nodded and said, "Okay, Mommy. I will be kind to everyone."
```

## friends_before_dark / focused / Model-Y-CPT-SFT-v2

- Words: 66.7% (oak, kind)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 2.9%

```text
Once upon a time, there was a kind girl named Lily. She loved to play outside in the sun. One day, she found a small oak tree in her backyard. She picked it up and felt happy.

But then, she saw a big tree. The tree was sad. Lily wanted to help the tree, so she asked her friend, the wise old owl. The owl said, "I can help you find your home."

Lily was happy to help the tree. She went to the tree and found a big tree. The tree was warm and warm. The tree was happy and thanked Lily.

From that day on, Lily and the wise owl became good friends. They played together in the sun and had lots of fun. The moral of the story is that we should always help others when they need it.
```

## named_lantern_crossing / greedy / Model-Y-Direct-SFT-v2

- Words: 66.7% (bridge, careful)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 8.2%

```text
Once upon a time, there was a little girl named Mia. She loved to play in the bridge. One day, she saw a bridge in the bridge. She wanted to cross it, but it was too high.

Mia asked her mom, "Can I cross the bridge?" Her mom said, "Yes, but be careful." Mia was careful and climbed on the bridge. She was careful not to slip and fall.

Mia was careful not to slip. She climbed up the bridge and reached the bridge. She was careful not to fall. She was careful not to fall. Mia was happy. She learned that it was important to be careful when she cross the bridge.
```

## named_lantern_crossing / greedy / Model-Y-CPT-SFT-v2

- Words: 33.3% (bridge)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 3.3%

```text
Once upon a time, there was a little girl named Mia. She loved to climb trees and play in the park. One day, she saw a bridge in the bridge. She wanted to cross it, but she was scared.

Mia asked her mom, "Can I cross the bridge?" Her mom said, "Yes, you can cross the bridge." Mia was happy and started to climb the bridge. She climbed and climbed until she reached the bridge.

When Mia reached the bridge, she saw a bridge. She was so excited. She climbed on the bridge and reached the bridge. She was so happy to see the bridge. She climbed the bridge and climbed the bridge. She felt proud of herself for being brave and helping her.
```

## named_lantern_crossing / focused / Model-Y-Direct-SFT-v2

- Words: 33.3% (bridge)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 0.8%

```text
Once upon a time, there was a little girl named Mia. She loved to go on adventures with her friends. One day, she went to the park with her mom. They saw a bridge and decided to cross it.

Mia's mom said, "Let's cross the bridge together!" Mia was excited and asked, "What is cross?" Her mom said, "It's a bridge. We can cross the bridge." Mia was happy and said, "Yes, let's cross the bridge!"

They walked and walked until they reached the other side. Mia said, "Look, there is a bridge!" Her mom said, "Yes, it is a bridge. It's very pretty." Mia and her mom sat on the bridge and enjoyed the view. They had a lot of fun on the bridge.
```

## named_lantern_crossing / focused / Model-Y-CPT-SFT-v2

- Words: 66.7% (bridge, careful)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 0.8%

```text
Once upon a time, there was a little girl named Mia. She loved to walk in the river with her mom. One day, they saw a bridge in the river. Mia wanted to cross it, but her mom said no.

Mia was sad and didn't know what to do. She asked her mom to help her cross the bridge. Her mom said, "Don't worry, Mia. I will help you cross the bridge."

Mia was very careful and followed the bridge. She climbed the bridge and reached the bridge. Suddenly, a river appeared and started to cross. Mia was scared, but she remembered the bridge. She started to cross the bridge and soon she was back in the bridge. Mia was happy she helped the bridge and continued to walk with her mom.
```

## lost_kite / greedy / Model-Y-Direct-SFT-v2

- Words: 33.3% (kite)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 2.7%

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a kite in the sky. She wanted to fly it, but it was too high.

Lily asked her mom, "Can you help me get the kite?"

Her mom said, "Yes, I can help you."

Lily and Ben went to the park to get the kite. They found a kite and put it on the ground. Lily was happy to help her kite.

But then, something unexpected happened. The kite started to fly away. Lily and Ben were sad. They looked everywhere for the kite, but they couldn't find it.

Lily's mom said, "Don't worry, we can find the kite." They looked and looked, but they couldn't find it. Then, they saw a big kite in the sky. Lily and Ben were so happy. They hugged each other and said, "
```

## lost_kite / greedy / Model-Y-CPT-SFT-v2

- Words: 33.3% (kite)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 1.2%

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a kite in the sky. She wanted to fly it, but she was scared.

Lily asked her mom, "Can I fly the kite?"

Her mom said, "Yes, you can. But be careful, it's very cold."

Lily was happy to have her kite back. She ran to the park and played with it. She was happy to have a new kite.
```

## lost_kite / focused / Model-Y-Direct-SFT-v2

- Words: 33.3% (kite)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 6.1%

```text
Once upon a time, there was a little girl named Lily. She loved to fly her kite in the sky. One day, she saw a kite in the sky. She wanted to catch it, but her kite was too fast.

Lily asked her mom, "Can you help me fly my kite?"

Her mom said, "Yes, I can help you."

Lily ran to the kite and tried to fly it. She was very careful not to lose it. She was very careful not to lose it.

After a few days, Lily's kite was gone. She was very sad. But her kite helped her fly. Lily was happy again. She said, "Thank you, mommy! You are a good friend."
```

## lost_kite / focused / Model-Y-CPT-SFT-v2

- Words: 33.3% (kite)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 8.1%

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big kite in the sky. She wanted to fly it, but it was too high.

Lily asked her mom, "Can I fly the kite?" Her mom said, "Yes, but be careful. It's too high."

Lily was sad to fly the kite. She asked her mom, "Can we go home now?" Her mom said, "Yes, we can go home."

They went to the park and saw a big kite. Lily said, "Wow, we can fly the kite!" Her mom said, "Yes, let's go to the park!" They went to the park and found the kite. Lily was happy to have her kite back.
```

## puppy_in_rain / greedy / Model-Y-Direct-SFT-v2

- Words: 33.3% (umbrella)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 4.5%

```text
Once upon a time, there was a little girl named Emma. She was three years old and loved to play outside. One day, Emma and Jack were playing in the park. Emma was playing with her umbrella when she saw a puddle. She wanted to jump in it, but Jack said no.

Emma said, "No, I want to play in the puddle!"

Jack said, "But I want to play in the puddle!"

Emma said, "No, you can't play in the puddle. It's too wet and it's too wet."

But Jack didn't listen. He kept playing in the puddle and it was very wet. Emma said, "I'm sorry, Jack. I didn't mean to hurt you."

Jack said, "It's okay, Emma. I'm glad you're safe."

Emma smiled and said, "I'm glad you're okay, Jack. I'm glad you're safe."
```

## puppy_in_rain / greedy / Model-Y-CPT-SFT-v2

- Words: 33.3% (umbrella)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 0.0%

```text
Once upon a time, there was a little girl named Emma. She was three years old and loved to play outside. One day, Emma and Jack were playing in the park. Emma was playing with her umbrella and she was having so much fun.

Suddenly, Emma heard a loud noise. She looked around and saw a big puddle. She was scared and ran to her mom.

"Mommy, what's that?" Emma asked.

"It's a puddle," her mom replied.

"It's a puddle, Emma," Emma said.

"Yes, it's a puddle," Jack said.

"Let's go inside and get some water," Emma said.

So, Emma and Jack went inside. Emma was so happy to be safe and sound.

"Thank you for playing with me," Emma said.

"You're welcome," Jack said.

"I'm glad you like it," Emma said.
```

## puppy_in_rain / focused / Model-Y-Direct-SFT-v2

- Words: 33.3% (umbrella)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 16.6%

```text
Once upon a time, there was a little girl named Emma. She was very wet and she was very wet. She had a umbrella that was wet and wet.

One day, Emma and Jack were playing in the rain. Emma said, "Let's go inside and get a umbrella!" Emma said, "OK!"

So, Emma and Jack went outside to play. Emma said, "Let's go inside and get some water."

So, Emma and Jack went inside and had a fun day playing in the rain. They laughed and had a great time.

When it was time to go inside, Emma said, "Let's go inside and play!"

So, Emma and Jack went inside to have a warm, warm blanket. Emma said, "Let's go inside and have some fun!"

So, Emma and Jack went inside and had a warm, warm blanket. They were happy and dry, and they played together every day.
```

## puppy_in_rain / focused / Model-Y-CPT-SFT-v2

- Words: 66.7% (puppy, umbrella)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 3.5%

```text
Once upon a time, there was a little puppy named Emma. Emma loved to play outside in the rain. One day, Emma was playing with her umbrella when she saw a big puddle. She wanted to go inside and get her umbrella.

 Emma ran to the puddle and started to splash. She splashed and splashed, but she couldn't find her umbrella. She started to cry.

"Why are you so wet?" asked Emma.

"I lost my umbrella," said Emma.

"Don't worry," said Emma. "Let's find it together."

So Emma and Jack went to the house. It was a big puddle of water. Emma was so happy! She ran to the puddle and splashed in the puddles.

"Thank you for helping me!" said Emma.

"You're welcome," said Emma. "Let's go inside and get you a umbrella."

So Emma and Jack went inside the house. They played games
```

## broken_robot / greedy / Model-Y-Direct-SFT-v2

- Words: 66.7% (robot, patient)
- Summary terms: 75.0%
- Features: 0.0%
- Entities: 0.0%
- Repeated 4-grams: 53.5%

```text
Once upon a time, there was a patient robot. He lived in a small house with a big, red robot. The robot was very patient and waited for the robot to fix his toy robot.

One day, a patient robot came to the house. The robot was very patient and waited for the robot to fix. The robot was very patient and waited for the robot to fix.

The robot was very patient and waited for the robot to fix its toy. The robot was very patient and waited for the robot to fix its toy. Finally, the robot was able to fix the robot's toy robot.

The robot was very happy and thanked the robot. The robot was very patient and waited for the robot to fix its toy. The robot was very patient and waited for the robot to fix its toy. The robot was very patient and waited for the robot to fix its toy.
```

## broken_robot / greedy / Model-Y-CPT-SFT-v2

- Words: 66.7% (robot, patient)
- Summary terms: 100.0%
- Features: 0.0%
- Entities: 50.0%
- Repeated 4-grams: 41.6%

```text
Once upon a time, there was a patient robot named Leo. Leo loved to play with his toy robot. He had a toy robot named Leo. Leo was very patient and always helped his robot.

One day, Leo's robot came to visit him. Leo was very patient and waited for Leo to fix his toy robot. Leo was very patient and waited for Leo to fix his toy robot.

After a while, Leo's robot came to visit. Leo was very patient and waited for Leo to fix his robot. Leo was very patient and waited for Leo to fix his robot. Finally, Leo's robot came to visit. Leo was very patient and waited for Leo to fix his robot.

The robot was very patient and waited for Leo to fix his robot. Leo was very patient and waited for Leo to fix his robot. Finally, the robot was fixed and the robot was fixed. Leo was very happy and thanked Leo for his help. From that day on, Leo
```

## broken_robot / focused / Model-Y-Direct-SFT-v2

- Words: 100.0% (robot, broken, patient)
- Summary terms: 100.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 0.8%

```text
Once upon a time, there was a patient robot named Leo. Leo lived in a big house with his family. They liked to play with their toy robot.

One day, Leo's mom said, "Leo, we are going to repair your robot toy. It will be fun." Leo was very excited. He wanted to help his mom fix the robot toy.

They went to the kitchen and found some tools. Leo started to repair the robot. He worked hard to fix it. He fixed the robot's toy car. Leo was very patient. He waited for the robot to fix.

Soon, the robot's broken toy car was fixed. Leo was so happy. He thanked his mom and they went to the toy toy. They played with the robot all day.
```

## broken_robot / focused / Model-Y-CPT-SFT-v2

- Words: 100.0% (robot, broken, patient)
- Summary terms: 100.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 5.0%

```text
Once upon a time, there was a patient robot named Leo. Leo was a patient robot. He loved to fix things. One day, he saw a broken toy robot. It was broken and broken. Leo wanted to fix it.

Leo asked his mom for help. His mom said, "Let's fix the robot." They worked together to fix the robot. They worked hard to fix it. The robot was very patient.

After a while, the robot was fixed. Leo was very happy. He said, "Thank you, robot! I can fix it!" The robot was happy. They fixed the robot's broken toy. They were very happy.
```

## garden_surprise / greedy / Model-Y-Direct-SFT-v2

- Words: 66.7% (garden, seed)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: 0.0%
- Repeated 4-grams: 12.6%

```text
Once upon a time, there was a little girl named Lily. She loved to plant flowers in the garden. One day, she found a big seed in the garden. She picked it up and put it in her pocket.

As she was walking, she saw a big, green flower. "Wow, that flower is so big!" she said. "Can I pick it?"

"Sure, Lily," said her mom. "But be careful, it's very big and strong."

Lily picked up the seed and put it in her pocket. She kept it in her pocket and kept it in her pocket.

As she was walking home, she saw a big, scary flower. "What's that?" she asked her mom. "It's a big, green flower," her mom replied. "It's just a flower, Lily. It's very big and strong."

Lily was happy to see the flower grow. She picked it up and put
```

## garden_surprise / greedy / Model-Y-CPT-SFT-v2

- Words: 66.7% (garden, seed)
- Summary terms: 50.0%
- Features: 0.0%
- Entities: 0.0%
- Repeated 4-grams: 15.8%

```text
Once upon a time, there was a little girl named Lily. She loved to plant seeds in her garden. One day, she found a big seed in the ground. She was so excited to plant it.

Lily picked up the seed and started to plant it in the ground. She watered it every day and watched it grow. Suddenly, the seed started to grow. It was so big that it could grow big and strong.

Lily was so happy to see the seed grow. She watered it every day and watched it grow. She watered it every day and watched it grow. It was so big and strong.

One day, Lily saw a big seed in the ground. She picked it up and put it in her garden. She watered it and waited for it to grow. Suddenly, the seed started to grow. It grew bigger and bigger. Lily was so happy and proud of her plant.

From that day on, Lily knew that even though she was
```

## garden_surprise / focused / Model-Y-Direct-SFT-v2

- Words: 66.7% (garden, seed)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: 0.0%
- Repeated 4-grams: 2.0%

```text
Once upon a time, there was a little girl named Lily. She loved to plant flowers in the garden. One day, she saw a big seed on the ground. She picked it up and put it in her pocket. 

As she was walking, she saw a big, juicy flower. She picked it up and showed it to her mom. Her mom said, "Wow, Lily! You did a great job picking flowers!" 

Lily was happy that she could plant the flower and make a big difference. She showed her mom the plant and said, "Look, Mommy! I found a big seed!" Her mom smiled and said, "That's a great idea, Lily. Let's plant it in our garden." 

Lily and her mom planted the plant and planted it in the garden. They watered it every day and it grew bigger and bigger. They were very happy and proud of their garden. The end.
```

## garden_surprise / focused / Model-Y-CPT-SFT-v2

- Words: 66.7% (garden, seed)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 0.0%
- Repeated 4-grams: 0.7%

```text
Once upon a time, there was a little girl named Lily. She loved to plant flowers in the garden. One day, she saw a big plant that was very tall. 

"Wow, that is a big seed!" said Lily. 

"Yes, it is!" said her mom. 

Lily took the seed home and planted it in the garden. She watered it every day and waited for it to grow. 

One day, Lily's mom asked her to plant a plant in the garden. Lily was happy to help and started to plant it in the garden. 

But then, something unexpected happened. The plant started to grow bigger and bigger. Lily was very sad. 

"I'm sorry, Mommy," said Lily. 

Her mom hugged her and said, "It's okay, Lily. We can plant a big plant together." 

Lily learned that sometimes, things can be big and strong. And she also learned that it's important
```
