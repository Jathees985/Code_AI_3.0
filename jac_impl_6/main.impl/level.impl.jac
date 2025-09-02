import sys;

impl level.run_game {
    # print("Walker Entry ->", self.level_id);

    if self.played == False {
        if visitor.current_level != self.game_level {
            visitor.current_level = self.game_level;
        }
        
        # Check if fine-tuned model exists to decide map generation strategy
        if os.path.exists("./tinyllama-rpg-finetuned") {
            # Fine-tuned model exists - generate AI-powered maps
            print(":-: Using AI-powered map generation for Level", self.game_level);
            try {
                # Initialize LevelManager if not already done
                if not hasattr(visitor.manager, 'current_model') {
                    visitor.manager.init_llm();
                }
                
                # Generate new level and map using AI
                if self.game_level == 1 {
                    visitor.manager.current_level = 0;
                } else {
                    visitor.manager.current_level = self.game_level - 1;
                }
                
                # Get AI-generated level data
                level_data = visitor.manager.get_next_level();
                generated_level = level_data[0];
                generated_map_obj = level_data[1];
                
                # Convert to game format
                llm_map = visitor.manager.get_map(generated_map_obj);
                self.level_config.map = llm_map;
                
                print(":-: AI-generated map loaded for Level", self.game_level);
                
            } except Exception as e {
                print(":-: Error generating AI map:", str(e));
                print(":-: Using default map for Level", self.game_level);
                # Fallback to default map
                self.level_config.map = [
                    'BBBBBBBBBBBBBBBBBBBB',
                    'B...E..............B',
                    'B.......B..........B', 
                    'B....BBBB..........B',
                    'B..................B',
                    'B..................B',
                    'B.........P........B',
                    'B..................B',
                    'B.............E....B',
                    'B..................B',
                    'B..................B',
                    'B.........B........B',
                    'B.........B........B',
                    'B.........B........B',
                    'BBBBBBBBBBBBBBBBBBBB'
                ];
            }
        } else {
            # No fine-tuned model - use static map for replay
            print(":-: No fine-tuned model found. Using static map for Level", self.game_level);
            self.level_config.map = [
                'BBBBBBBBBBBBBBBBBBBB',
                'B...E..............B',
                'B.......B..........B', 
                'B....BBBB..........B',
                'B..................B',
                'B..................B',
                'B.........P........B',
                'B..................B',
                'B.............E....B',
                'B..................B',
                'B..................B',
                'B.........B........B',
                'B.........B........B',
                'B.........B........B',
                'BBBBBBBBBBBBBBBBBBBB'
            ];
        }
        
        visitor.g.GameMap.map = self.level_config.map;
        visitor.g.new();
        print(":-: Playing Level :", self.game_level, "| Level ID :",  self.level_id, "| Played :", str(self.played));
        start_time = time.time();
        visitor.g.main();
        end_time = time.time();
        visitor.last_level_id = self.level_id;
        if visitor.g.won == True{
            self.level_time = end_time - start_time;
            if visitor.manager.prev_levels {
                visitor.manager.prev_levels[-1].time = self.level_time;
            }
            
            # Check if we have fine-tuned model for progression logic
            if os.path.exists("./tinyllama-rpg-finetuned") {
                # AI model exists - proceed to next level with new AI-generated map
                print(":-: Level", self.game_level, "completed! Generating AI-powered Level", str(self.game_level+1));
                visitor.g.game_won();
                self.played = True;
                visitor.g.won = False;
                visitor.fwd_dir = True;
                new_ID = str(self.game_level+1) + "_" + str(random.randint(1000, 9000));
                self +>:play:level_id=new_ID:+> level(game_level = self.game_level+1, level_id = new_ID);
                visit [->:play:level_id==new_ID:->];
            } else {
                # No AI model - replay the same level
                print(":-: Level", self.game_level, "completed! No AI model found - replaying same level");
                visitor.g.game_won();
                self.played = False;  # Mark as not played so it can be replayed
                visitor.g.won = False;
                # Stay on the same level
                visit [<-:play:<-];
            }
        } else {
            # Level failed - always replay the same level with same map
            print(":-: Level", self.game_level, "failed! Replaying same level");
            visitor.g.game_over();
            self.played = False;  # Allow replay
            visitor.g.won = False;
            visitor.fwd_dir = False;
            visit [<-:play:<-];
        }
    } elif visitor.fwd_dir == False{
        # Handle replay scenarios
        if os.path.exists("./tinyllama-rpg-finetuned") {
            # AI model exists - can proceed to next level
            new_ID = str(self.game_level+1) + "_" + str(random.randint(1000, 9000));
            print(":-: Moving to next Level :", str(self.game_level+1), "| New Level ID :", new_ID);
            visitor.fwd_dir = True;
            self +>:play:level_id=new_ID:+> level(game_level = self.game_level+1, level_id = new_ID);
            visit [->:play:level_id==new_ID:->];
        } else {
            # No AI model - replay current level
            print(":-: No AI model found. Replaying Level :", self.game_level);
            self.played = False;  # Reset to allow replay
            visitor.fwd_dir = True;
            visit [<-:play:<-];
        }
    } else {
        print(":-: Visiting Level :", self.game_level, "| Level ID :",  self.level_id, "| Played :", str(self.played));
        visit [<--];
    }
}

impl level.exit_game {
    if visitor.g.running == False {
        pygame.quit();
        sys.exit();
        disengage;
    }
}