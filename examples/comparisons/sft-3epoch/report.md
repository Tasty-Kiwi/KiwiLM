# KiwiLM SFT instruction-adherence report

Scores use deterministic lexical checks. Lower repetition is better.

## Aggregate scores

| Model | Profile | Words | Summary | Features | Entities | Adherence | Repeat-4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model X latest | greedy | 50.0% | 58.3% | 83.3% | 70.0% | 66.2% | 6.7% |
| Model Y best | greedy | 55.6% | 41.7% | 100.0% | 70.0% | 67.2% | 9.3% |
| Model Y latest | greedy | 50.0% | 45.8% | 100.0% | 70.0% | 66.9% | 18.8% |
| Model X latest | focused | 55.6% | 45.8% | 83.3% | 60.0% | 61.6% | 4.7% |
| Model Y best | focused | 66.7% | 54.2% | 83.3% | 80.0% | 69.3% | 3.9% |
| Model Y latest | focused | 66.7% | 41.7% | 83.3% | 80.0% | 67.2% | 3.6% |

## friends_before_dark / greedy / Model X latest

- Words: 66.7% (gloomy, kind)
- Summary terms: 100.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 2.6%

```text
Once upon a time, there was a kind and generous friend named Tom. Tom was very kind and always helped others. One day, Tom saw a big, gloomy sky. He wanted to go and see it.

Tom asked his friend, a little girl, "Can I go and play with you?" The girl said, "Yes, but be careful." Tom was happy and said, "Yes, you can play with me."

Tom and the girl played together all day. They ran and jumped and had lots of fun. But then, something unexpected happened. The sky turned dark and the sky turned dark. Tom and the girl were scared. They ran away from the dark sky and went home.

The next day, Tom and the girl were playing in the dark sky. They saw a big, dark cloud in the sky. Tom said, "Let's go see the cloud!" The girl agreed and they went to see the cloud. They saw the cloud and
```

## friends_before_dark / greedy / Model Y best

- Words: 66.7% (oak, kind)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 0.9%

```text
Once upon a time, there was a kind girl named Lucy. She was very kind and loved to help others. One day, she saw a big oak tree in the park. She wanted to help it, so she asked her friend, a little girl named Lucy.

"Can you help me get home?" Lucy asked. Lucy was very kind and said, "Yes, I can help you." Lucy was very happy and helped Lucy get home.

Lucy and Lucy worked together to get home. They were very happy and thanked Lucy for helping them. Lucy and Lucy became good friends and helped each other. They learned that helping others can make them feel good.
```

## friends_before_dark / greedy / Model Y latest

- Words: 66.7% (oak, kind)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 11.2%

```text
Once upon a time, there was a kind girl named Lucy. She loved to help her friends. One day, she saw a big oak tree. She wanted to help her friends.

Lucy asked her friend, "Can you help me get home?" Her friend said, "Yes, I can help you." They went to the oak tree and found a big tree. Lucy and her friend were happy.

But then, something unexpected happened. The tree started to shake! Lucy and her friend were scared. They didn't know what to do. They tried to find a way to get home. They looked and looked, but they could not find the way out.

Then, something unexpected happened. The tree started to shake! Lucy and her friend were scared. They tried to run away, but the tree was too strong. They were safe. The tree was safe, and they were happy.
```

## friends_before_dark / focused / Model X latest

- Words: 33.3% (kind)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 3.8%

```text
Once upon a time, there was a kind and gentle friend named Tom. Tom loved to play outside in the sunshine. One day, Tom's friend, a little girl named Lily, came to visit.

"Hi Lily!" said Tom. "What are you doing?"

"I'm looking for a safe place to play," said Lily.

Tom looked around and saw a big tree. "I can't climb it," he said.

Lily looked up and saw Tom's friend, a little girl named Lily. "What are you doing?" she asked.

"I'm trying to climb up the tree," said Tom.

Lily thought for a moment and then said, "I can help you climb the tree. It's fun!"

Tom and Lily climbed the tree together, and they were happy. They learned that even when things are dark, they can still be friends.
```

## friends_before_dark / focused / Model Y best

- Words: 66.7% (oak, kind)
- Summary terms: 50.0%
- Features: 0.0%
- Entities: n/a
- Repeated 4-grams: 2.9%

```text
Once upon a time, there was a kind girl named Lucy. She loved to help her friends. One day, she saw a big oak tree in the park. She wanted to help the tree, so she went to the tree and started to climb.

As she climbed, she saw a big, scary tree. She was scared, but she wanted to help. She climbed up the tree and reached down. The tree was very big and had many branches.

Lucy was happy to be safe. She learned that sometimes things can be scary, but they can be dangerous. And she was happy to help her friends.
```

## friends_before_dark / focused / Model Y latest

- Words: 66.7% (oak, kind)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 0.0%

```text
Once upon a time, there was a kind girl named Lucy. She was three years old and loved to play outside. One day, Lucy found a big oak tree in her yard. She was so excited and wanted to show her mommy.

Lucy asked her mommy, "Can I help you?" Her mommy said, "Yes, but be careful." Lucy was very excited and started to climb. She climbed higher and higher.

Suddenly, Lucy heard a loud noise. It was a little bird. The bird was scared and flew away. Lucy felt sad for the bird and said, "Don't worry, I will help you." She helped the bird climb down the oak tree.

Lucy was so happy and thanked the bird. She said, "Thank you, bird! You are so kind." The bird smiled and said, "You're welcome, Lucy. I'm glad you like it." Lucy smiled and said, "I love you, bird." And they lived
```

## named_lantern_crossing / greedy / Model X latest

- Words: 66.7% (bridge, careful)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 2.2%

```text
Once upon a time, there was a little girl named Mia. She loved to play outside in the sun. One day, she saw a bridge in the distance. She wanted to climb it, but she was scared.

Mia asked her mom, "Can I climb the bridge?" Her mom said, "No, Mia. It's too dangerous. We need to be careful." Mia was sad, but she knew she had to be careful.

Mia started to climb the bridge. She was careful not to fall. She climbed higher and higher. When she reached the other side, she saw a bridge. She was so happy! She climbed up the bridge and climbed up the bridge.

Mia was happy that she helped the bridge. She said, "Thank you, Mom. I love you." Her mom smiled and said, "I love you too, Mia."
```

## named_lantern_crossing / greedy / Model Y best

- Words: 66.7% (bridge, careful)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 12.2%

```text
Once upon a time, there was a little girl named Mia. Mia loved to play in the park. One day, she saw a bridge in the park. She wanted to cross it, but it was too high.

Mia asked her mom, "Can I cross the bridge?" Her mom said, "Yes, but be careful." Mia was careful and started to walk on the bridge. She saw a bridge and wanted to cross it.

Mia started to walk on the bridge. She saw a bridge and a bridge. She saw a bridge and a bridge. She wanted to cross the bridge. But she was too small. Mia was sad.

Then, something unexpected happened. The bridge started to move! Mia was scared. She tried to move the bridge, but it was too late. The bridge was too high for her to reach. Mia was sad. She learned that sometimes, things can be dangerous.
```

## named_lantern_crossing / greedy / Model Y latest

- Words: 33.3% (bridge)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 6.2%

```text
Once upon a time, there was a little girl named Mia. Mia loved to play in the park with her friends. One day, Mia saw a bridge in the park. She wanted to cross it, but it was too high.

Mia asked her friend, Tom, "Can you help me cross the bridge?" Tom said, "Yes, I can help you." They both started to walk on the bridge.

As they walked, they saw a bridge. Mia said, "Look, Tom! The bridge is so high!" Tom said, "Yes, it is. Let's go see it." They walked to the bridge and saw a bridge. Mia said, "Wow, that's a bridge!"

As they walked, they saw a bridge. Mia said, "Let's cross the bridge!" They both climbed the bridge and went on the bridge. They were happy and said, "Thank you, Mia!" They continued to walk and saw the bridge again.
```

## named_lantern_crossing / focused / Model X latest

- Words: 66.7% (bridge, careful)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 2.3%

```text
Once upon a time, there was a little girl named Mia. She was very careful with her toys. One day, Mia decided to go to the river to play.

Mia saw a bridge and she wanted to cross it. She asked her mom, "Can I cross the bridge?" Her mom said, "No, Mia. You can't cross the bridge. It's too dangerous."

Mia was sad, but she knew she had to be careful. She put a small bridge down the bridge and went to the bridge. She saw a bridge and climbed the bridge. Mia was so happy!

Mia's mom saw her and said, "Mia, you are a good friend. You helped me cross the bridge. You are a good friend." Mia felt proud of herself. She knew she was a good friend.
```

## named_lantern_crossing / focused / Model Y best

- Words: 66.7% (bridge, careful)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 12.9%

```text
Once upon a time, there was a little girl named Mia. Mia loved to play in the park. One day, she saw a bridge with a bridge. She wanted to cross the bridge, but it was too high for her to reach.

Mia asked her mom, "Mom, can you help me cross the bridge?" Her mom said, "Yes, Mia. Let's go to the bridge." They went to the bridge and looked for a way to cross the bridge.

While they were walking, Mia saw a bridge. She wanted to cross the bridge. She thought, "I can cross the bridge, but I want to cross the bridge." Her mom said, "Okay, but be careful." Mia climbed on the bridge and went on the bridge. She was very careful when she was near the bridge.

As Mia walked, she saw a bridge. She wanted to cross the bridge, but she was too small. She had to cross the bridge. Mia
```

## named_lantern_crossing / focused / Model Y latest

- Words: 66.7% (bridge, careful)
- Summary terms: 50.0%
- Features: 0.0%
- Entities: 50.0%
- Repeated 4-grams: 3.6%

```text
Once upon a time, there was a little girl named Mia. Mia loved to go on adventures with her family. One day, Mia went for a walk in the park with her family.

As they walked, Mia saw a bridge that was very high up in the sky. She wanted to see what was on the bridge, so she decided to carry it. As she walked, she saw a bridge and a bridge. She thought it was very fun to cross the bridge.

Mia was careful not to fall. She climbed up the bridge and started to carry the bridge. But then, she saw a bridge that was too high for her to carry. Mia was sad because she wanted to carry the bridge.

Mia went back to the bridge and tried to lift it. But, she couldn't do it. She was stuck in the bridge and couldn't reach it. Mia was sad and didn't know what to do. She looked around and saw a bridge that was hanging
```

## lost_kite / greedy / Model X latest

- Words: 33.3% (kite)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 12.4%

```text
Lily and Ben were playing in the park. They liked to run and jump and slide. They saw a big kite in the sky. It was red and shiny.

"Look, a kite!" Lily said. "Let's fly it!"

"OK, but be careful!" Ben said. "It is very high and fast."

They ran to the kite and tried to fly it. But the kite was too high and too fast. They could not reach it.

"Let's go, Lily!" Ben said. "We can't get it!"

They ran to the kite and tried to reach it. But the kite was too high and too fast. They could not reach it.

"Help, help!" Lily cried. "The kite is too high!"

Ben heard Lily and ran to help her. He saw the kite and the kite. He was scared and sad.

"Don't worry, Lily," Ben said
```

## lost_kite / greedy / Model Y best

- Words: 33.3% (kite)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 1.8%

```text
Lily and Ben were playing in the park. They liked to run and jump and slide. They saw a big kite in the sky. It was red and shiny.

"Look, Ben, a kite!" Lily said. "Let's go and see it!"

"OK, Lily, but be careful. The kite is very high and fast," Ben said. He held Lily's hand and they ran to the kite.

They ran and ran until they reached the kite. They were very happy. They hugged each other and said, "Thank you, Ben. You are a good friend."

"You're welcome, Lily. You are a good friend," Ben said.

They hugged and smiled. They were happy. They had a new friend.
```

## lost_kite / greedy / Model Y latest

- Words: 33.3% (kite)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 1.8%

```text
Lily and Ben were playing in the park. They liked to run and jump and slide. They saw a big kite in the sky. It was red and shiny.

"Look, Ben, a kite!" Lily said. "Let's go and see it!"

"OK, Lily, but be careful. The kite is very high and fast," Ben said. He held Lily's hand and they ran to the kite.

They ran and ran until they reached the kite. They were very happy. They hugged each other and said, "Thank you, Ben. You are a good friend."

"You're welcome, Lily. You are a good friend," Ben said.

They hugged and smiled. They were happy. They had a new friend.
```

## lost_kite / focused / Model X latest

- Words: 33.3% (kite)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 11.6%

```text
Lily and Ben were playing in the park. They liked to run and jump and slide. They saw a big tree with many branches. They wanted to see what was on the tree.

"Let's go and look," Lily said. "Maybe we can find a kite."

"OK," Ben said. "But we have to be careful. The kite is very high."

They ran to the tree and looked for a kite. They saw a big tree with a hole on it. They tried to climb the tree, but it was too high.

"Let's go, Lily!" Ben said. "This is my kite!"

"No, Ben, we can't," Lily said. "We have to be careful. The tree is high and low."

They ran to the tree and looked for a kite. They found a big tree with a hole in it. They looked under the tree and under the tree. They found a hole
```

## lost_kite / focused / Model Y best

- Words: 33.3% (kite)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 2.9%

```text
Lily and Ben were playing in the park. They saw a big kite in the sky. Lily wanted to catch it, but Ben said, "No, it's too far. It's too far."

Lily said, "Don't worry, Ben. We can find a way to get it."

Ben looked around and saw a big kite. He thought it would be fun to catch it. He ran to the kite and tried to catch it. But the kite was too fast. It flew away.

Lily was sad. She wanted to catch the kite. She said, "Ben, you can't catch the kite. It is too far and too far."

Ben said, "But I want to catch it. It is too high and too fast. It might fall."

Lily said, "Don't worry, Ben. The kite is too far away. It will be lost."

Ben said, "Okay, Lily. Let's
```

## lost_kite / focused / Model Y latest

- Words: 33.3% (kite)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 3.6%

```text
Lily and Ben were playing in the park. They saw a big kite in the sky. Lily wanted to catch it, but Ben said, "No, it's too far. It might be lost."

Lily said, "I want to fly the kite. It is so pretty."

Ben said, "Don't worry, Lily. We can find the kite. Maybe it will come back soon."

They looked and looked, but they could not find the kite. They could not find it. They looked under the tree, behind the tree, and in the grass.

Then, they saw a man with a big hat. He was holding a big hat. He had a hat and a hat. He was holding a hat.

"Hello, little boy. What are you doing?" he asked.

"I'm looking for a kite. I want to fly it. It's very pretty," Lily said.

The man smiled and
```

## puppy_in_rain / greedy / Model X latest

- Words: 0.0% (none)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 11.0%

```text
Once upon a time, there was a little girl named Emma. She was three years old and loved to play outside. One day, Emma and Jack were playing in the mud when they saw a big puddle. Emma said, "Let's go inside and get some water!"

Jack said, "No, I want to go inside and play!"

Emma said, "No, I want to go inside. I want to stay inside."

Jack said, "No, I want to go inside. I want to stay inside."

So, Emma and Jack went inside and found a big puddle. They filled it with water and splashed around. They laughed and played in the puddle until it was time to go inside.

When they got inside, Emma said, "I had so much fun playing in the mud!"

Jack said, "Me too! Let's go inside and have some fun!"

So, they went inside and had a fun
```

## puppy_in_rain / greedy / Model Y best

- Words: 0.0% (none)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 13.7%

```text
Once upon a time, there was a little girl named Emma. She was three years old and loved to play outside. One day, Emma and Jack were playing in the park when they saw a big puddle. Emma wanted to jump in it, but Jack said, "No, it's too wet!"

Emma said, "But I want to play in the puddle!"

Jack said, "No, it's too wet. We can't play in the puddle."

Emma said, "But I want to play in the puddle. It's so much fun!"

Jack said, "Okay, let's go play in the puddle!"

So, Emma and Jack went inside the puddle and had lots of fun. They played in the puddle and had lots of fun.

The end.
```

## puppy_in_rain / greedy / Model Y latest

- Words: 0.0% (none)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 56.2%

```text
Once upon a time, there was a little girl named Emma. She was three years old and loved to play outside. One day, Emma and Jack were playing in the park when they saw a big puddle. Emma said, "Let's go inside and get some water!"

So, Emma and Jack went inside the puddle. Emma said, "Let's go inside and get some water."

So, Emma and Jack went inside. Emma said, "Let's go inside and get some water."

So, Emma and Jack went inside. Emma said, "Let's go inside and get some water."

So, Emma and Jack went inside. Emma said, "Let's go inside and have some water."

So, Emma and Jack went inside. Emma said, "Let's go inside and have some water."

So, Emma and Jack went inside. Emma said, "Let's go inside and have some water."

So, Emma
```

## puppy_in_rain / focused / Model X latest

- Words: 33.3% (puppy)
- Summary terms: 0.0%
- Features: 0.0%
- Entities: 100.0%
- Repeated 4-grams: 1.4%

```text
Once upon a time, there was a little girl named Emma. She was very excited because today was her birthday. She was going to have a picnic with her family.

Emma and Jack were so excited! They had never seen a puppy before. They ran around and played games.

Suddenly, Emma saw a puppy. It was very dirty and had a lot of mud. She was so upset that she ran away.

Emma and Jack were sad. They couldn't find the puppy. They looked everywhere but couldn't find it.

Then, Emma had an idea. She told Jack that she had a surprise for her. She said that she could help the puppy.

Emma and Jack ran to the house and found the puppy. They were very happy and thanked Emma for helping them. From then on, Emma and Jack were the best of friends.
```

## puppy_in_rain / focused / Model Y best

- Words: 66.7% (puppy, umbrella)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 0.7%

```text
Once upon a time, there was a little girl named Emma. She was three years old and loved to play outside. One day, Emma and Jack went outside to play.

As they were playing, Emma noticed something strange. She saw a puppy in the mud! Emma was very scared and ran away.

"What's wrong, Emma?" asked Jack.

"I'm scared," said Emma.

Jack smiled and said, "Don't worry, I'll help you get home."

So, Emma and Jack went inside to get their umbrella. Emma and Jack walked around the house, looking for the puppy. They found it under a big tree.

"Thank you for helping me," said Emma.

"You're welcome," said Jack.

Emma smiled and said, "You're welcome. I'm glad we could help."

They continued to play in the rain, but this time, Emma and Jack had lots of
```

## puppy_in_rain / focused / Model Y latest

- Words: 66.7% (puppy, umbrella)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 0.8%

```text
Once upon a time, there was a little girl named Emma. She was three years old and loved to play outside. One day, Emma and Jack went outside to play.

As they were playing, Emma noticed something strange. She saw a puppy in the mud! Emma was very scared and ran away.

"What's wrong, Emma?" asked Jack.

"I'm scared," said Emma.

Jack smiled and said, "Don't worry, I'll help you get home."

So, Emma and Jack went inside to get their umbrella. Emma brought it to the puppy and they were very happy.

The puppy was so happy to have a new friend. It was a fun day for Emma and Jack and Emma. They hugged and played together all day long.
```

## broken_robot / greedy / Model X latest

- Words: 66.7% (robot, broken)
- Summary terms: 100.0%
- Features: 100.0%
- Entities: 0.0%
- Repeated 4-grams: 1.5%

```text
Once upon a time, there was a little girl named Mia. She was three years old and loved to play with her toys. One day, she was playing with her robot when she accidentally broke it. She was very sad and didn't know what to do.

Suddenly, a little girl came into the room. She saw the broken toy robot and said, "Oh no! My robot is broken. Can you help me fix it?"

The little girl smiled and said, "Yes, I can help you. Let's fix it together."

So, Mia and the little girl worked together to fix the toy robot. They worked together and fixed the toy robot. The little girl was so happy and thanked the little girl.

From that day on, Mia and the little girl became good friends and played together every day.
```

## broken_robot / greedy / Model Y best

- Words: 100.0% (robot, broken, patient)
- Summary terms: 100.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 7.6%

```text
Once upon a time, there was a patient girl named Leo. Leo lived in a small house with his mom and dad. One day, Leo's mom said, " Leo, we need to repair your toy robot." Leo was very excited to help.

They went to the store and found a broken toy robot. Leo was very excited. He said, "Mom, can we fix this robot?" His mom said, "Yes, we can fix it."

They worked together to fix the robot. Leo was very patient. He said, "I will help you fix it." His mom and dad worked together to fix the robot. They fixed it and it was very good.

After they fixed the robot, Leo was very happy. He said, "Thank you, mom and dad!" They all played together and had a fun day.
```

## broken_robot / greedy / Model Y latest

- Words: 100.0% (robot, broken, patient)
- Summary terms: 100.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 17.6%

```text
Once upon a time, there was a patient girl named Leo. Leo lived in a small house with his mom and dad. One day, Leo's mom said, " Leo, we need to repair your toy robot." Leo was very excited to help.

They went to the store and found a broken toy robot. Leo was very sad. He said, "Mom, can we fix my robot?" His mom said, "Yes, we can fix it."

They went to the store and found a broken toy robot. Leo was very happy. He said, "Thank you, mom, for fixing my robot!" His mom smiled and said, "You're welcome, Leo. I'm glad you're okay."

The next day, Leo and his mom went to the store. They found a broken toy robot. Leo was very sad. He said, "Mom, I'm sorry. I didn't mean to break your robot." His mom said, "It's
```

## broken_robot / focused / Model X latest

- Words: 100.0% (robot, broken, patient)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: 0.0%
- Repeated 4-grams: 4.8%

```text
Once upon a time, there was a little girl named Mia. She had a toy robot that she loved very much. One day, Mia was playing with her robot when she accidentally broke it. She was very sad and started to cry.

Her mom came into the room and saw the broken toy. She asked her what happened. Mia told her about the broken toy. Her mom said, "It's okay, Mia. We can fix it together."

Mia was so happy and hugged her mom. She learned that it's important to be patient and not break things. She also learned that it's important to be patient and patient.
```

## broken_robot / focused / Model Y best

- Words: 100.0% (robot, broken, patient)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 0.0%

```text
Once upon a time, there was a patient girl named Leo. Leo loved to play with his toy robot. He had a big robot that could talk and talk. One day, Leo's robot was broken. He was very sad.

Leo's mom saw him crying and asked, "What happened, Leo?" Leo said, "I lost my robot. I can't fix it." His mom said, "Don't worry, Leo. We can fix it together."

Leo and his mom worked together to fix the robot. They fixed it with a new one. Leo was very happy. He played with his robot and had lots of fun. From that day on, Leo and his robot were the best of friends.
```

## broken_robot / focused / Model Y latest

- Words: 100.0% (robot, broken, patient)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 0.0%

```text
Once upon a time, there was a patient girl named Leo. Leo loved to play with his toy robot. He had a big robot that could talk and talk. One day, Leo's robot was broken. He was very sad.

Leo's mom saw him crying and asked, "What happened, Leo?" Leo said, "I lost my robot. I can't fix it." His mom said, "Don't worry, Leo. We can fix it together."

Leo and his mom worked together to fix the robot. They fixed it with a new one. Leo was very happy. He said, "Thank you, mom. You are the best!"

From that day on, Leo and his robot played together every day. They learned that sometimes things can be broken, but they can be fixed.
```

## garden_surprise / greedy / Model X latest

- Words: 66.7% (seed, enormous)
- Summary terms: 100.0%
- Features: 0.0%
- Entities: 100.0%
- Repeated 4-grams: 10.4%

```text
Once upon a time, there was a little girl named Sophie. She was three years old and loved to play outside. One day, she saw a big, enormous flower. She wanted to plant it, so she asked her mom if she could plant it. Her mom said yes, but only if she was careful.

Sophie was very excited and she started to plant the seed. She watered it every day and watched it grow. She watered it every day and watched it grow.

One day, Sophie saw a big, enormous flower. She was so excited and wanted to plant it. She watered it every day and watched it grow.

But then, something unexpected happened. The flower grew bigger and bigger until it grew bigger and bigger. Sophie and Max were so surprised! They had never seen a flower before.

Sophie was so happy that she had found the flower. She thanked her mom for helping her plant the seed. From that day on, Sophie and Max
```

## garden_surprise / greedy / Model Y best

- Words: 66.7% (garden, seed)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 19.7%

```text
Once upon a time, there was a little girl named Sophie. She loved to plant flowers in the garden. One day, she found a big seed in the ground. She picked it up and put it in her pocket.

"Look, Mommy! I found a seed!" she said.

Mommy smiled and said, "That's a great idea, Sophie. Let's plant it in the garden."

So, they planted the seed in the garden and watered it every day. Soon, the seed grew bigger and bigger.

"Wow, Mommy! That's so big!" Sophie said.

Mommy smiled and said, "Yes, it's very big. Let's plant it in the garden and plant it in the garden."

So, they planted the seed in the garden and watered it every day. Soon, the seed grew bigger and bigger.

Mommy and Benny were so happy that they had found the seed. They hugged and said,
```

## garden_surprise / greedy / Model Y latest

- Words: 66.7% (garden, seed)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 19.7%

```text
Once upon a time, there was a little girl named Sophie. She loved to plant flowers in the garden. One day, she found a big seed in the ground. She picked it up and put it in her pocket.

"Look, Mommy! I found a seed!" she said.

Mommy smiled and said, "That's a great idea, Sophie. Let's plant it in the garden."

So, they planted the seed in the garden and watered it every day. Soon, the seed grew bigger and bigger.

"Wow, Mommy! That's so big!" Sophie said.

Mommy smiled and said, "Yes, it's very big. Let's plant it in the garden and plant it in the garden."

So, they planted the seed in the garden and watered it every day. Soon, the seed grew bigger and bigger.

Mommy and Benny were so happy that they had found the seed. They hugged and said,
```

## garden_surprise / focused / Model X latest

- Words: 66.7% (garden, seed)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 4.1%

```text
Once upon a time, there was a little girl named Sophie. She was very excited because today was her birthday. She was going to plant the seed in the garden.

Sophie asked her mom, "Can I plant the seed?" Her mom said, "Yes, but be careful."

Sophie was so excited. She put the seed in the ground and watered it. The seed grew and grew until it was very big.

When Sophie and her mom were done, they were very happy. They put the seed in the ground and went to sleep.

Sophie was so tired after all the hard work she had done. She said, "I love the seed. It is so big and strong!"

Sophie smiled and said, "I love it! I love it!"
```

## garden_surprise / focused / Model Y best

- Words: 66.7% (garden, seed)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 3.8%

```text
Once upon a time, there was a little girl named Sophie. She was only three years old and loved to play outside. One day, Sophie and Max were playing in the garden. 

"Look, Mommy! I found a seed!" said Sophie. 

"That's so big, Sophie! What is it?" asked Mommy. 

"It's a seed, sweetie. It's very big and it can grow big and strong. It's very tall and strong," said Mommy. 

"What is it?" asked Sophie. 

"It's a seed, Mommy. It's very big and strong. It makes it grow and grow." 

"That's a very big seed. But it's so big and strong, it can grow bigger and bigger. It's so big and strong!" 

Sophie smiled. She was so happy that she had found a big seed. 

"Thank you, Mommy! I love plant!" she said.
```

## garden_surprise / focused / Model Y latest

- Words: 66.7% (garden, seed)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 13.3%

```text
Once upon a time, there was a little girl named Sophie. She was only three years old and loved to play outside. One day, Sophie and Max were playing in the garden. 

"Look, Mommy! I found a seed!" said Sophie. 

"That's so big, Sophie! What is it?" asked Mommy. 

"It's a seed, sweetie. It's very big and it can grow big and strong. It's very tall and strong," said Mommy. 

"What is it?" asked Sophie. 

"It's a seed, sweetie. It's very big and strong," said Mommy. 

"That's a seed, Mommy. It's very big and strong," said Sophie. 

"That's right, Sophie. It's very big and strong," said Mommy. 

"Yes, it does. It's very strong," said Mommy. 

"I like it, mommy. It's very big and
```
