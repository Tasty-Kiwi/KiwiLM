# KiwiLM SFT instruction-adherence report

Scores use deterministic lexical checks. Lower repetition is better.

## Aggregate scores

| Model | Profile | Words | Summary | Features | Entities | Adherence | Repeat-4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Model X best | greedy | 38.9% | 41.7% | 83.3% | 80.0% | 58.0% | 13.6% |
| Model X latest | greedy | 50.0% | 58.3% | 83.3% | 70.0% | 66.2% | 6.7% |
| Model Y | greedy | 33.3% | 58.3% | 100.0% | 40.0% | 58.7% | 4.9% |
| Model X best | focused | 27.8% | 45.8% | 100.0% | 70.0% | 60.4% | 8.5% |
| Model X latest | focused | 55.6% | 45.8% | 83.3% | 60.0% | 61.6% | 4.7% |
| Model Y | focused | 33.3% | 41.7% | 50.0% | 40.0% | 40.3% | 5.6% |

## friends_before_dark / greedy / Model X best

- Words: 0.0% (none)
- Summary terms: 25.0%
- Features: 0.0%
- Entities: n/a
- Repeated 4-grams: 37.7%

```text
Once upon a time, there was a little girl named Jane. She was three years old and loved to explore. One day, she decided to go on an adventure. She walked and walked until she came to a big, dark cave. She was scared, but she wanted to go inside.

Jane walked into the cave and saw a big, dark cave. She was scared, but she wanted to go inside. She walked inside and saw a big, dark cave. She was scared, but she was brave.

Suddenly, she heard a loud noise. It was a big, dark cave. Jane was scared, but she was brave. She stepped inside and saw a big, dark cave. She was scared, but she was brave.

Jane was brave and she stepped inside. She saw a big, dark cave. She was scared, but she was brave. She stepped inside and saw a big, dark cave. She was scared, but she was brave.
```

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

## friends_before_dark / greedy / Model Y

- Words: 0.0% (none)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 2.0%

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she saw a big tree with lots of leaves. She wanted to climb the tree, but it was too high. 

Lily's friend, Timmy, came over to play. "Can you help me climb the tree?" asked Lily. "Yes, I can help you," said Timmy. They climbed up the tree and climbed up the tree. 

As they were climbing, they saw a big tree. "Wow, that tree is so pretty!" said Lily. "Yes, it is," said Timmy. "But we need to be careful." 

Lily and Timmy climbed the tree and climbed the tree. They were so happy to be back home. "Thank you for helping me, Timmy," said Lily. "You're welcome, Lily. I'm glad we could help." And they continued to play in the sun, happy to have helped each other
```

## friends_before_dark / focused / Model X best

- Words: 0.0% (none)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: n/a
- Repeated 4-grams: 4.4%

```text
Once upon a time, there was a little girl named Mary. Mary was three years old and loved to play outside. One day, she found a big, dark cloud in the sky. She was so excited! She wanted to explore it, so she ran to the cloud.

When she got there, she saw a big, dark cloud. It was so big and scary. She was scared and started to cry.

Her friend, a little girl, came over and asked, "What's wrong, Mary?"

"I'm scared," Mary said.

The little girl said, "Don't be scared, I'm here to help you."

Mary smiled and said, "I'm here to help you. I'm here to help you."

The little girl was so happy that she hugged Mary and said, "Thank you for helping me. I'm glad I could help."

Mary smiled and said, "You're welcome!"
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

## friends_before_dark / focused / Model Y

- Words: 0.0% (none)
- Summary terms: 50.0%
- Features: 0.0%
- Entities: n/a
- Repeated 4-grams: 2.8%

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sun. One day, she found a big tree with lots of leaves. She wanted to climb the tree to see what was on the other side.

As she climbed up the tree, she saw a little bird with a broken wing. The bird was sad because it couldn't fly. Lily felt sorry for the bird and decided to help it. She climbed up the tree and found a big rock. The rock was so happy and thanked the bird.

Lily felt happy that she could help the bird. She learned that even though a little bird can be lost, it can still be a good friend. And she learned that even though things don't go as planned, they can still be fun and make things happen.
```

## named_lantern_crossing / greedy / Model X best

- Words: 66.7% (bridge, careful)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 0.8%

```text
Once upon a time, there was a little girl named Mia. She loved to play outside in the sun. One day, she saw a bridge in the distance. It was very tall and had lots of trees. Mia wanted to climb the bridge, but she was scared.

Mia asked her mom, "Mom, can I climb the bridge?" Her mom said, "No, Mia. You can't climb the bridge. It's too dangerous." Mia was sad, but she knew she had to be careful.

Mia started to climb the bridge. She climbed and climbed until she reached the top. She looked down and saw the bridge. She was so happy. Mia said, "Thank you, Mia! You are so brave!" Her mom smiled and said, "You're welcome, Mia. You're a good friend."
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

## named_lantern_crossing / greedy / Model Y

- Words: 66.7% (bridge, careful)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 5.7%

```text
Once upon a time, there was a little girl named Mia. She loved to play in the park with her friends. One day, Mia saw a big bridge in the park. She wanted to climb it, but it was too high.

Mia's friend, a little girl named Mia, came to play. Mia saw Mia and said, "I want to climb the bridge!" Mia said, "No, Mia! I want to climb the bridge!" Mia was sad, but she wanted to help.

Mia and Mia started to climb the bridge. They climbed and climbed until they reached the other side. Mia was so happy to have her friend back. They played on the bridge all day long. Mia and Mia were very careful not to fall down.
```

## named_lantern_crossing / focused / Model X best

- Words: 66.7% (bridge, careful)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 1.6%

```text
Once upon a time, there was a little girl named Mia. She was three years old and loved to explore. One day, Mia went out to the bridge to see what was on the other side. 

Mia saw a bridge and wanted to climb it. She asked her mom if she could climb it. Her mom said yes, but only if Mia was careful. 

Mia started to climb the bridge. She was careful not to get too close. She was careful not to fall.

Mia climbed up the bridge and reached the other side. She felt very proud of herself. She showed her mom and dad and said, "Look, Mia! I did it! I did it!" 

Mia's mom smiled and said, "That was a great idea, Mia. I'm proud of you."
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

## named_lantern_crossing / focused / Model Y

- Words: 66.7% (bridge, careful)
- Summary terms: 50.0%
- Features: 0.0%
- Entities: 50.0%
- Repeated 4-grams: 11.0%

```text
Once upon a time, there was a little girl named Mia. She had a big bridge that she loved to carry. One day, Mia found a bridge in the park. It was very tall and had lots of trees and flowers. Mia wanted to carry the bridge, so she started to carry it.

As Mia walked, she saw a big bridge. It was very tall and had many colors. Mia wanted to carry the bridge, but she was too small to carry it. She tried to carry the bridge, but it was too heavy. Mia tried and tried, but she could not carry it.

Then, Mia had an idea. She decided to carry the bridge with her. She carried the bridge and went on her way. She was very careful not to carry it. When she got to the other side, she saw a bridge. She wanted to climb it, but she was too small.

Mia tried to carry the bridge, but it was too heavy.
```

## lost_kite / greedy / Model X best

- Words: 0.0% (none)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 26.4%

```text
Lily and Ben were playing in the park. They liked to run and jump and slide. They saw a big tree with many branches. They wanted to climb the tree and see what was on the other side.

"Let's go, Lily!" Ben said. "Maybe we can find a way to get to the top."

"OK, Ben," Lily said. "But we have to be careful. The tree is high and the tree is high."

They ran to the tree and looked around. They saw a big tree with many branches. They wanted to climb the tree.

"Let's go, Lily," Ben said. "Maybe we can find a way to get to the top."

They climbed the tree and looked around. They saw a big tree with many branches. They climbed the tree and looked at the tree. It was very high.

"Wow, look at the tree!" Lily said. "It is so high and
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

## lost_kite / greedy / Model Y

- Words: 33.3% (kite)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 14.7%

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a kite in the sky. She wanted to catch it, but it was too high.

Lily asked her mom, "Can you help me get the kite?" Her mom said, "Yes, I can help you." They went to the park and found a kite. Lily was so happy and said, "Thank you, Mommy!"

But then, Lily saw a big kite in the sky. She said, "Mommy, can we get the kite?" Her mom said, "Yes, we can get it." They went to the park and found the kite. Lily was so happy and said, "Thank you, Mommy!"
```

## lost_kite / focused / Model X best

- Words: 0.0% (none)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 26.9%

```text
Lily and Ben were playing in the park. They liked to run and jump and slide. They saw a big tree with many branches. They wanted to see what was on the tree.

"Let's go and look," Lily said.

"OK, but be careful. The tree is very high," Ben said.

They ran to the tree and looked around. They saw a big tree with a hole in it. They wanted to see what was inside.

"Let's go and see," Lily said.

"OK, but be careful. It might be very dark," Ben said.

They ran to the tree and looked around. They saw a big tree with a hole in it. They saw a hole in the ground. They saw a hole in the ground. They saw a hole in the ground.

"What is in the hole?" Lily asked.

"I don't know. Maybe there is a hole in the
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

## lost_kite / focused / Model Y

- Words: 33.3% (kite)
- Summary terms: 0.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 3.0%

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big kite in the sky. She wanted to fly it, but her mom said it was too high.

Lily asked her mom, "Can we go to the kite?" Her mom said, "Yes, we can go to the kite." Lily was excited to go on the kite.

They went to the kite and got the kite. Lily was so happy! She said, "Thank you, Mommy!" Then they went home and played with the kite. Lily was happy to be back.
```

## puppy_in_rain / greedy / Model X best

- Words: 33.3% (puppy)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 4.7%

```text
Once upon a time, there was a little girl named Emma. She was very excited because she was going to the park with her mommy.

When they got to the park, Emma saw a puppy. She was so excited! She ran over to the puppy and said, "Hi puppy! Can I pet you?"

The puppy wagged its tail and licked Emma's face. Emma was so happy to have a new friend. She said, "Thank you, Max! You are so kind!"

Max and Lucy played together all day. They ran and jumped and laughed. Emma was so happy to have a new friend.

At the end of the day, Emma and Jack went home. Emma said, "Thank you for helping me, Max. You are a very kind and helpful puppy."

The end.
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

## puppy_in_rain / greedy / Model Y

- Words: 0.0% (none)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 2.8%

```text
Once upon a time, there was a little girl named Emma. She was three years old and loved to play outside. One day, Emma and Jack were playing in the park. Emma was playing with a ball and Jack wanted to catch it.

Emma said, "Let's go play!"

Jack said, "Yes, let's go!"

So they ran to the park and started to play. They ran and ran until they found a big puddle. Emma said, "Let's go inside and have some fun!"

Jack said, "Yes, let's go inside!"

So they went inside and had a fun day playing in the rain. They laughed and played until it was time to go home.
```

## puppy_in_rain / focused / Model X best

- Words: 0.0% (none)
- Summary terms: 25.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 6.6%

```text
Once upon a time, there was a little girl named Emma. She was three years old and loved to play outside. One day, Emma wanted to go outside and play. She asked her mom if she could go outside and play. Her mom said yes, but only if Emma was careful.

So Emma and Jack went outside and looked around. They saw a big puddle and decided to jump in it. Emma and Jack ran around and laughed. They had so much fun playing together.

Suddenly, Emma started to feel cold. She wanted to go outside and play, but she was too scared to go outside. She said to Jack, "Don't worry, I'll help you."

So Emma and Jack went outside and found a big puddle. They filled it with water and water. Emma and Jack laughed and played in the puddle. They had so much fun that they didn't even notice the rain stopped.
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

## puppy_in_rain / focused / Model Y

- Words: 0.0% (none)
- Summary terms: 0.0%
- Features: 0.0%
- Entities: 100.0%
- Repeated 4-grams: 10.5%

```text
Once upon a time, there was a little girl named Emma. She was three years old and loved to play outside. One day, Emma went for a walk in the park. She saw a big puddle and wanted to jump in it. She ran over to the puddle and jumped in.

The puddle was so soft and fun! Emma ran over to the puddle and jumped in. She laughed and played in the mud. She laughed and had so much fun.

Then, Emma saw something else. It was a big puddle! She wanted to jump in it. She jumped in and jumped in. The puddle was so big and filled with water. Emma laughed and splashed around in the puddle.

Emma and Jack laughed and played in the puddle. They had a great time. The end.
```

## broken_robot / greedy / Model X best

- Words: 66.7% (robot, patient)
- Summary terms: 100.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 4.8%

```text
Once upon a time, there was a little girl named Leo. Leo was very patient and always did what his mom said. One day, Leo was playing with his toy robot when he accidentally broke it. He was very sad and started to cry.

Suddenly, Leo's mom came into the room and saw what happened. She said, "Oh no, Leo! You broke my toy robot! You broke it!" Leo was very sad and started to cry.

His mom hugged him and said, "It's okay, Leo. We can fix it together. We can fix it together." Leo was so happy and they worked together to fix the toy robot.

The moral of the story is that it's important to be patient and help others when they need it.
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

## broken_robot / greedy / Model Y

- Words: 66.7% (robot, patient)
- Summary terms: 100.0%
- Features: 100.0%
- Entities: 0.0%
- Repeated 4-grams: 1.4%

```text
Once upon a time, there was a little girl named Mia. Mia loved to play with her robot. She had a toy robot that she loved very much. One day, Mia's robot broke. Mia was sad.

Mia's mom saw her sad face and asked, "Why are you sad, Mia?" Mia said, "I lost my robot. I can't find it." Mia's mom said, "Don't worry, Mia. We can find it together."

Mia and her mom looked under the bed. They looked under the bed, in the closet, and in the closet. Finally, they found the robot under the bed. Mia was so happy! She hugged her robot and said, "Thank you, robot! You are the best!"

Mia and her mom played with the robot all day. They had lots of fun. Mia learned that sometimes, things can be fixed, but it's important to be patient and help others.
```

## broken_robot / focused / Model X best

- Words: 33.3% (robot)
- Summary terms: 100.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 1.4%

```text
Once upon a time, there was a little girl named Leo. Leo was three years old and loved to play with his toy robot. One day, Leo was playing with his toy robot when he accidentally broke it. He was very sad and started to cry.

Suddenly, Leo heard a loud noise. He looked around and saw a big robot. The robot was stuck in a big box. Leo knew he had to help the robot. He carefully opened the box and found a small toy robot!

The robot was so happy and thanked Leo. He hugged his robot and said, "Thank you for helping me fix my robot. He is so kind and helpful."

From that day on, Leo and the robot became best friends. They played together every day, and Leo was no longer sad. And they all lived happily ever after.
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

## broken_robot / focused / Model Y

- Words: 66.7% (robot, broken)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: 0.0%
- Repeated 4-grams: 2.1%

```text
Once upon a time, there was a little girl named Mia. Mia had a toy robot that she loved very much. One day, Mia was playing with her robot when she accidentally broke it. She was very sad and didn't know what to do.

Mia's mom saw her crying and asked, "What happened, Mia?" Mia replied, "My robot is broken. I can't fix it." Her mom said, "Don't worry, Mia. We can fix it together." They went to the store and found a new robot. Mia was so happy and said, "Thank you, mom!"

Mia and her mom went to the store to buy a new toy robot. They bought it and put it in a big box. Mia was so happy to have her robot back. She said, "Thank you, mom! I love my robot!" And they both went home, happy and content.
```

## garden_surprise / greedy / Model X best

- Words: 66.7% (garden, enormous)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 100.0%
- Repeated 4-grams: 7.1%

```text
Once upon a time, there was a little girl named Sophie. She was three years old and loved to play outside. One day, she saw a big, enormous flower. She wanted to pick it up, so she picked it up and put it in her pocket.

Sophie asked her mom, "Can I pick the flower?" Her mom said, "No, you can't pick it. It's too big for you."

Sophie was sad, but she knew she had to be careful. She picked up the flower and put it in her pocket. She said, "I will pick the flower and put it in my pocket."

So Sophie and Max picked the flower and put it in the garden. They were very happy and thanked the flower for helping them.

The end.
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

## garden_surprise / greedy / Model Y

- Words: 33.3% (garden)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 0.0%
- Repeated 4-grams: 2.6%

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the garden. One day, she saw a big plant in the garden. She wanted to plant it, but it was too big for her. 

Lily asked her mom, "Can you help me plant the plant?" 

Her mom said, "Yes, I can help you. Let's plant the plant together." 

So, they planted the plant in the garden and planted it in the garden. The plant grew bigger and bigger until it was as big as the plant. 

Lily was so happy that she could plant the plant and plant it in the garden. She said, "Thank you, Mommy! I love plant!"
```

## garden_surprise / focused / Model X best

- Words: 66.7% (garden, seed)
- Summary terms: 50.0%
- Features: 100.0%
- Entities: 50.0%
- Repeated 4-grams: 10.5%

```text
Once upon a time, there was a little girl named Sophie. She was very excited because today was a big garden. She wanted to plant something special.

Sophie asked her mom, "Can I plant a seed?" Her mom said, "Yes, you can plant it in the garden."

Sophie was very excited. She carefully plant the seed in the ground and watered it every day. Then she watered the seed and watched it grow.

One day, Sophie was very excited. She wanted to plant the seed in the garden. She found a big seed and watered it every day.

But when she tried to plant the seed, she saw a big seed in the ground. She was very curious. She wanted to plant the seed.

So, she asked her mom for help. Her mom said, "Yes, you can plant the seed. It will grow big and strong."

Sophie was so happy. She thanked her mom and said, "
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

## garden_surprise / focused / Model Y

- Words: 33.3% (garden)
- Summary terms: 75.0%
- Features: 100.0%
- Entities: 0.0%
- Repeated 4-grams: 4.2%

```text
Once upon a time, there was a little girl named Lily. She loved to plant flowers in the garden. One day, she saw a big flower and wanted to plant it. 

Lily said, "Mommy, look! The flower is so big and strong. Can I pick it?" 

Mommy said, "Sure, but be careful. The flower is very big and can break easily." 

Lily was happy to pick the flower and put it in the garden. She watered it every day and watched it grow. 

One day, while she was picking flowers, she saw a big flower. She said, "Mommy, look! A big flower! Can you pick it?" 

Mommy said, "Yes, Lily. I can pick it. It's very big and strong." 

Lily picked up the flower and put it in her pocket. She said, "I'm sorry, Mommy. I didn't know it was a flower. I just wanted to
```
