# KiwiLM SFT instruction-adherence report

Scores use deterministic lexical checks. Lower repetition is better.

## Aggregate scores

| Model | Profile | Words | Summary | Features | Entities | Adherence | Repeat-4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model Y SFT v2 | greedy | 55.6% | 45.8% | 83.3% | 50.0% | 59.6% | 15.0% |
| Model Y SFT v2 | focused | 55.6% | 50.0% | 100.0% | 50.0% | 65.2% | 5.0% |

## friends_before_dark / greedy / Model Y SFT v2

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

## friends_before_dark / focused / Model Y SFT v2

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

## named_lantern_crossing / greedy / Model Y SFT v2

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

## named_lantern_crossing / focused / Model Y SFT v2

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

## lost_kite / greedy / Model Y SFT v2

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

## lost_kite / focused / Model Y SFT v2

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

## puppy_in_rain / greedy / Model Y SFT v2

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

## puppy_in_rain / focused / Model Y SFT v2

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

## broken_robot / greedy / Model Y SFT v2

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

## broken_robot / focused / Model Y SFT v2

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

## garden_surprise / greedy / Model Y SFT v2

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

## garden_surprise / focused / Model Y SFT v2

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
