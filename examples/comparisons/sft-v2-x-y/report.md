# KiwiLM SFT instruction-adherence report

Scores use deterministic lexical checks. Lower repetition is better.

## Aggregate scores

| Model | Profile | Words | Summary | Features | Entities | Adherence | Repeat-4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model X v2 best | greedy | 38.9% | 29.2% | 100.0% | 60.0% | 56.6% | 12.5% |
| Model X v2 latest | greedy | 44.4% | 29.2% | 100.0% | 60.0% | 58.0% | 12.5% |
| Model Y v2 | greedy | 55.6% | 45.8% | 83.3% | 50.0% | 59.6% | 15.0% |
| Model X v2 best | focused | 50.0% | 41.7% | 83.3% | 40.0% | 54.6% | 7.6% |
| Model X v2 latest | focused | 50.0% | 58.3% | 100.0% | 40.0% | 63.0% | 4.0% |
| Model Y v2 | focused | 55.6% | 50.0% | 100.0% | 50.0% | 65.2% | 5.0% |

## friends_before_dark / greedy / Model X v2 best

- Words: 0.0% (none)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 28.5%

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she saw a big tree with a big tree. She wanted to climb it, but it was too high for her to reach.

Lily asked her friend, Timmy, "Can you help me climb the tree?" Timmy said, "Sure, I can help you climb the tree." They climbed the tree and climbed the tree. They climbed the tree and climbed the tree.

As they climbed the tree, they saw a big tree. The tree was very tall and had many branches. Lily said, "Look, Timmy! The tree is so pretty!" Timmy said, "Yes, it is. Let's climb it." They climbed the tree and climbed the tree.

As they climbed the tree, they saw a big tree. The tree was very tall and had many branches. Lily said, "Look, Timmy! The tree is so pretty!"
```

## friends_before_dark / greedy / Model X v2 latest

- Words: 0.0% (none)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 28.5%

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she saw a big tree with a big tree. She wanted to climb it, but it was too high for her to reach.

Lily asked her friend, Timmy, "Can you help me climb the tree?" Timmy said, "Sure, I can help you climb the tree." They climbed the tree and climbed the tree. They climbed the tree and climbed the tree.

As they climbed the tree, they saw a big tree. The tree was very tall and had many branches. Lily said, "Look, Timmy! The tree is so pretty!" Timmy said, "Yes, it is. Let's climb it." They climbed the tree and climbed the tree.

As they climbed the tree, they saw a big tree. The tree was very tall and had many branches. Lily said, "Look, Timmy! The tree is so pretty!"
```

## friends_before_dark / greedy / Model Y v2

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

## friends_before_dark / focused / Model X v2 best

- Words: 33.3% (kind)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 0.0%

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she found a big tree with lots of leaves. She was very happy and wanted to climb it.

Lily said, "I want to climb the tree, but I am too small to reach the top."

Her friend, Timmy, said, "I can help you, Lily. Let's climb the tree and see what's on the top."

So, they climbed the tree and climbed the tree. They climbed higher and higher until they reached the top. Lily was so happy to be outside and play with her friends. She said, "Thank you, Timmy! You are so kind."
```

## friends_before_dark / focused / Model X v2 latest

- Words: 33.3% (kind)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 0.0%

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she found a big tree with lots of leaves. She was very happy and wanted to climb it.

Lily said, "I want to climb the tree, but I am too small to reach the top."

Her friend, Timmy, said, "I can help you, Lily. Let's climb the tree and see what's on the top."

So, they climbed the tree and climbed the tree. They climbed higher and higher until they reached the top. Lily was so happy and said, "Thank you, Timmy! You are very kind."

From that day on, Lily and Timmy were the best of friends. They would climb the tree together and have fun together. And they lived happily ever after.
```

## friends_before_dark / focused / Model Y v2

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

## named_lantern_crossing / greedy / Model X v2 best

- Words: 66.7% (bridge, careful)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 13.0%

```text
Mia was a little girl who loved to climb. She was very careful and always kept her safe. One day, she saw a bridge in the distance. It was very tall and had a big bridge. Mia wanted to climb the bridge, but she was scared.

Mia asked her mom, "Mom, can I climb the bridge?" Her mom said, "No, Mia. The bridge is too high and too far. We need to be careful." Mia was sad, but she knew she had to be careful.

Mia climbed the bridge and climbed the bridge. She was careful not to fall. She climbed the bridge and got to the other side. She was happy and safe. She did not see the bridge that was going to fall. She did not see the bridge that was going to fall.

Mia was scared and sad. She did not want to fall. She wanted to climb the bridge. She did not want to fall. She wanted to climb the bridge
```

## named_lantern_crossing / greedy / Model X v2 latest

- Words: 66.7% (bridge, careful)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 13.0%

```text
Mia was a little girl who loved to climb. She was very careful and always kept her safe. One day, she saw a bridge in the distance. It was very tall and had a big bridge. Mia wanted to climb the bridge, but she was scared.

Mia asked her mom, "Mom, can I climb the bridge?" Her mom said, "No, Mia. The bridge is too high and too far. We need to be careful." Mia was sad, but she knew she had to be careful.

Mia climbed the bridge and climbed the bridge. She was careful not to fall. She climbed the bridge and got to the other side. She was happy and safe. She did not see the bridge that was going to fall. She did not see the bridge that was going to fall.

Mia was scared and sad. She did not want to fall. She wanted to climb the bridge. She did not want to fall. She wanted to climb the bridge
```

## named_lantern_crossing / greedy / Model Y v2

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

## named_lantern_crossing / focused / Model X v2 best

- Words: 66.7% (bridge, careful)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 12.0%

```text
Mia was very careful. She loved to climb the bridge. She would climb up the bridge and see what she could see. She was careful not to drop anything.

One day, Mia saw a bridge. It was very big and had a big bridge. She wanted to climb the bridge. She climbed up the bridge and reached the other side. She was careful not to fall.

She climbed the bridge. She reached the other side. She was careful not to fall. She did not see the bridge. She did not see the bridge. It was dark and deep.

Mia's mom heard her. She ran to the bridge. She saw Mia on the bridge. She was angry. She shouted at Mia. She said, "Mia, what are you doing? You are not allowed to climb the bridge. You are very naughty. You need to stay here."

Mia felt sorry. She did not mean to hurt Mia. She did not mean to make her
```

## named_lantern_crossing / focused / Model X v2 latest

- Words: 66.7% (bridge, careful)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 12.0%

```text
Mia was very careful. She loved to climb the bridge. She would climb up the bridge and see what she could see. She was careful not to drop anything.

One day, Mia saw a bridge. It was very big and had a big bridge. She wanted to climb the bridge. She climbed up the bridge and reached the other side. She was careful not to fall.

She climbed the bridge. She reached the other side. She was careful not to fall. She did not see the bridge. She did not see the bridge. It was dark and deep.

Mia's mom heard her. She ran to the bridge. She saw Mia on the bridge. She was angry. She shouted at Mia. She said, "Mia, what are you doing? You are not allowed to climb the bridge. You are very naughty. You need to stay here."

Mia felt sorry. She did not mean to hurt Mia. She did not mean to make her
```

## named_lantern_crossing / focused / Model Y v2

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

## lost_kite / greedy / Model X v2 best

- Words: 33.3% (kite)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 10.7%

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she saw a kite in the sky. She wanted to fly it, but she was scared.

Lily asked her mom, "Can I fly the kite?" Her mom said, "Yes, but be careful." Lily was happy to help her mom. They went to the park and saw a kite. The kite was red and shiny.

Lily and Ben played with the kite all day. They had so much fun. But then, Lily saw a kite in the sky. She said, "Look, Mommy! The kite is flying!" Her mom said, "Yes, it is. But be careful, it is very cold."

Lily and Ben played with the kite all day. They had so much fun. When it was time to go home, Lily said, "Thank you, Mommy. I love you." Her mom said, "I
```

## lost_kite / greedy / Model X v2 latest

- Words: 33.3% (kite)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 10.7%

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she saw a kite in the sky. She wanted to fly it, but she was scared.

Lily asked her mom, "Can I fly the kite?" Her mom said, "Yes, but be careful." Lily was happy to help her mom. They went to the park and saw a kite. The kite was red and shiny.

Lily and Ben played with the kite all day. They had so much fun. But then, Lily saw a kite in the sky. She said, "Look, Mommy! The kite is flying!" Her mom said, "Yes, it is. But be careful, it is very cold."

Lily and Ben played with the kite all day. They had so much fun. When it was time to go home, Lily said, "Thank you, Mommy. I love you." Her mom said, "I
```

## lost_kite / greedy / Model Y v2

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

## lost_kite / focused / Model X v2 best

- Words: 33.3% (kite)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 7.5%

```text
Once upon a time, there was a little girl named Lily. She loved to kite with her mommy. One day, they went to the park to play. Lily saw a big kite and wanted to fly it. She asked her mommy if she could fly it. Her mommy said yes and they went to the kite.

As they were flying, Lily saw a kite flying in the sky. She asked her mommy, "Can I fly the kite?" Her mommy said, "Sure, but be careful." Lily held the kite and flew it in the sky.

As they flew the kite, Lily saw a kite flying high in the sky. She said, "Mommy, can we fly the kite?" Her mommy said, "Sure, let's fly it together!" They flew the kite up high in the sky and the kite flew up and down. Lily was very happy and said, "Thank you, mommy!"
```

## lost_kite / focused / Model X v2 latest

- Words: 33.3% (kite)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 7.5%

```text
Once upon a time, there was a little girl named Lily. She loved to kite with her mommy. One day, they went to the park to play. Lily saw a big kite and wanted to fly it. She asked her mommy if she could fly it. Her mommy said yes and they went to the kite.

As they were flying, Lily saw a kite flying in the sky. She asked her mommy, "Can I fly the kite?" Her mommy said, "Sure, but be careful." Lily held the kite and flew it in the sky.

As they flew the kite, Lily saw a kite flying high in the sky. She said, "Mommy, can we fly the kite?" Her mommy said, "Sure, let's fly it together!" They flew the kite up high in the sky and the kite flew up in the sky. Lily was so happy and said, "Thank you, mommy!"
```

## lost_kite / focused / Model Y v2

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

## puppy_in_rain / greedy / Model X v2 best

- Words: 33.3% (umbrella)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 11.8%

```text
Once upon a time, there was a little girl named Emma. She was very dirty and had a big umbrella. She loved to play outside and run around in the mud. One day, Emma and Jack were playing in the mud when they saw a big puddle. Emma said, "Let's go inside and get wet!"

Jack said, "No, I want to go inside and play!"

So, Emma and Jack went inside and played in the mud. They had so much fun!

But then, Emma said, "I want to go inside now!"

Jack said, "No, I want to go inside now."

So, Emma and Jack went inside and had a fun day playing in the mud. They laughed and had a great time.
```

## puppy_in_rain / greedy / Model X v2 latest

- Words: 33.3% (umbrella)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 11.8%

```text
Once upon a time, there was a little girl named Emma. She was very dirty and had a big umbrella. She loved to play outside and run around in the mud. One day, Emma and Jack were playing in the mud when they saw a big puddle. Emma said, "Let's go inside and get wet!"

Jack said, "No, I want to go inside and play!"

So, Emma and Jack went inside and played in the mud. They had so much fun!

But then, Emma said, "I want to go inside now!"

Jack said, "No, I want to go inside now."

So, Emma and Jack went inside and had a fun day playing in the mud. They laughed and had a great time.
```

## puppy_in_rain / greedy / Model Y v2

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

## puppy_in_rain / focused / Model X v2 best

- Words: 33.3% (umbrella)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 2.5%

```text
Once upon a time, there was a little girl named Emma. She was three years old and loved to play outside. One day, she found a big puddle in the yard. It was so wet and slippery. Emma was sad and started to cry.

"Mommy, can you help me?" Emma asked.

"Sure, Emma. Let's go inside and get some dry clothes," her mommy said.

They went inside and Emma's mommy put the umbrella on the ground. Emma was so happy and thanked her mommy.

"Thank you for helping me," said Emma.

"You're welcome," said her mommy.

"You're welcome," said Emma.

They went inside and Emma's mommy gave her a warm blanket to rest. Emma felt happy and comfortable in her warm towel.
```

## puppy_in_rain / focused / Model X v2 latest

- Words: 33.3% (umbrella)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 0.0%

```text
Once upon a time, there was a little girl named Emma. She was three years old and loved to play outside. One day, she found a big puddle in the yard. It was so wet and slippery. Emma was sad and started to cry.

"Mommy, can you help me?" Emma asked.

"Sure, Emma. Let's go inside and get some dry clothes," her mommy said.

They went inside and Emma's mommy put the umbrella on the ground. Emma was so happy and thanked her mommy.

"Thank you for helping me," Emma said.

"You're welcome, Emma. I'm glad you're safe," her mommy said.

Emma smiled and hugged her mommy. She was happy to have her mommy back and the rain was gone.
```

## puppy_in_rain / focused / Model Y v2

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

## broken_robot / greedy / Model X v2 best

- Words: 33.3% (robot)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: 0.0%
- Repeated 4-grams: 1.8%

```text
Once upon a time, there was a little girl named Mia. She loved to play with her robot. One day, she saw a robot in the sky. The robot was very big and had a long tail. Mia wanted to fix the robot, so she asked her mom for help.

Her mom said, "Let's fix the robot and fix it." They worked together to fix the robot. They fixed the robot and the robot was fixed. The robot was very happy and thanked Mia.

After the robot fixed, Mia and her robot were very happy. They played with the robot all day long. They had a lot of fun. The robot was not sad anymore.
```

## broken_robot / greedy / Model X v2 latest

- Words: 66.7% (robot, broken)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: 0.0%
- Repeated 4-grams: 2.2%

```text
Once upon a time, there was a little girl named Mia. She loved to play with her robot. One day, she saw a robot in the sky. The robot was very big and had a long tail. Mia wanted to fix the robot, so she asked her mom for help.

Her mom said, "Let's fix the robot and fix it." They worked together to fix the robot. The robot was very happy. The robot was not broken anymore. Mia and her robot were happy. They played with the robot all day long.
```

## broken_robot / greedy / Model Y v2

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

## broken_robot / focused / Model X v2 best

- Words: 66.7% (robot, patient)
- Summary terms: 100.0%
- Features: 0.0%
- Entities: 0.0%
- Repeated 4-grams: 9.5%

```text
Once upon a time, there was a little girl named Mia. She was very patient and always did what her mom said. One day, Mia's mom told her to fix her toy robot. Mia was very patient and did not give up.

Mia was very patient and waited patiently. Finally, her mom came and fixed the toy robot. Mia was so happy and thanked her mom for helping her.

From that day on, Mia knew that if she worked hard, she could fix the toy robot. She knew that if she worked hard, she could still fix it.
```

## broken_robot / focused / Model X v2 latest

- Words: 66.7% (robot, patient)
- Summary terms: 100.0%
- Features: 100.0%
- Entities: 0.0%
- Repeated 4-grams: 2.3%

```text
Once upon a time, there was a little girl named Mia. She was three years old. One day, she was playing with her robot outside. She saw a robot in the sky and wanted to play with it.

Mia asked the robot, "Can I help you fix your toy robot?" The robot replied, "Yes, but be careful. It's very fragile." Mia was very happy.

The robot said, "Thank you, Mia. You are a good friend." Mia smiled and said, "You're welcome, little girl." She was very patient and kept helping the robot.

The robot and Mia played together all day. They had so much fun. When it was time to go home, Mia said, "Thank you for helping me fix my toy robot!" The robot smiled and said, "You're welcome, Mia. I'm glad you're okay."
```

## broken_robot / focused / Model Y v2

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

## garden_surprise / greedy / Model X v2 best

- Words: 66.7% (garden, seed)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 9.1%

```text
Once upon a time, there was a little girl named Sophie. She was three years old and loved to plant flowers. One day, she saw a big plant in the garden. It was very big and round. Sophie wanted to plant it, so she asked her mom for help.

Her mom said, "Yes, but be careful. The plant is big and strong."

Sophie and her mom planted the seed in the ground. It was very big and strong. Sophie and her mom watered the seed every day.

One day, Sophie and her mom went to the garden. Sophie saw the seed and said, "Mommy, look! The seed is so big!" Her mom smiled and said, "Yes, it is very big. Let's plant it in the ground."

So, Sophie and her mom planted the seed in the ground. They watered it and watered it every day. Sophie was so happy and proud of her plant. She said, "Look
```

## garden_surprise / greedy / Model X v2 latest

- Words: 66.7% (garden, seed)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 9.1%

```text
Once upon a time, there was a little girl named Sophie. She was three years old and loved to plant flowers. One day, she saw a big plant in the garden. It was very big and round. Sophie wanted to plant it, so she asked her mom for help.

Her mom said, "Yes, but be careful. The plant is big and strong."

Sophie and her mom planted the seed in the ground. It was very big and strong. Sophie and her mom watered the seed every day.

One day, Sophie and her mom went to the garden. Sophie saw the seed and said, "Mommy, look! The seed is so big!" Her mom smiled and said, "Yes, it is very big. Let's plant it in the ground."

So, Sophie and her mom planted the seed in the ground. They watered it and watered it every day. Sophie was so happy and proud of her plant. She said, "Look
```

## garden_surprise / greedy / Model Y v2

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

## garden_surprise / focused / Model X v2 best

- Words: 66.7% (garden, seed)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 14.1%

```text
Once upon a time, there was a little girl named Sophie. She was very small and very small. One day, Sophie said to her dad, "I want to plant the seed."

Her dad said, "I can plant the seed in the ground and make it grow big and strong."

Sophie and her dad went to the garden and planted the seed. They watered it every day and watched it grow.

One day, Sophie and her dad went to the garden. It was very big and tall. Sophie said, "Look at the plant, it is so big!"

Her dad said, "Yes, it is very big and strong. It is very big and strong."

Sophie and her dad went to the garden and found the seed. They watered it and watered it every day. When it was finished, Sophie and her dad were very happy. They had a big garden and a big garden.
```

## garden_surprise / focused / Model X v2 latest

- Words: 66.7% (seed, enormous)
- Summary terms: 100.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 2.1%

```text
Once upon a time, there was a little girl named Sophie. She was very small and very small. One day, Sophie said to her friend, "Let's plant a big flower!"

Sophie replied, "No, let's plant it!"

So they plant the seed in the ground. It was a big, enormous flower. Sophie said, "Look, Sophie! I found a big flower!"

Sophie said, "Wow, that's amazing! It's so big and big!"

So Sophie and Jack went to the flower. They watered the flower and it grew bigger and bigger. They were very happy because they had a big, enormous flower. 

The end.
```

## garden_surprise / focused / Model Y v2

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
