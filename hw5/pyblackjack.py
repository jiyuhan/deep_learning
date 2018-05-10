# import sys
import os
import random


class Blackjack():
    def __init__(self):
        self.deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] * 4
        self.funds = 100.0
        self.bet = 5
        self.max_plays = 10
        self.num_plays = 0
        self.dealer_hand = []
        self.player_hand = []
        print("There are", len(self.deck), "cards in the deck")

    def deal(self):
        hand = []
        for i in range(2):
            k = random.randint(0, 51)
            card = self.deck[k]
            if card == 11: card = "J"
            if card == 12: card = "Q"
            if card == 13: card = "K"
            if card == 14: card = "A"
            hand.append(card)
        return hand

    def game_over(self):
        self.num_plays += 1
        if self.num_plays <= self.max_plays:
            self.dealer_hand = self.deal()
            self.player_hand = self.deal()
            return False
        else:
            # print("Bye!")
            return True

    def total(self, hand):
        total, aces = 0, 0
        for card in hand:
            if card == "J" or card == "Q" or card == "K":
                total += 10
            elif card == "A":
                total += 11
                aces += 1
            else:
                total += card

        while total > 21 and aces > 0:
            total -= 10
            aces -= 1

        return total

    def hit(self, hand):
        # card = deck.pop()
        k = random.randint(0, 51)
        assert k < 52, (k, len(self.deck))
        card = self.deck[k]
        if card == 11: card = "J"
        if card == 12: card = "Q"
        if card == 13: card = "K"
        if card == 14: card = "A"
        hand.append(card)
        return hand

    def blackjack(self):
        if self.total(self.player_hand) == 21:
            self.print_results()
            print("Congratulations! You got a Blackjack!\n")
            self.funds += 1.5 * self.bet
            return True
        elif self.total(self.dealer_hand) == 21:
            self.print_results()
            print("Sorry, you lose. The dealer got a blackjack.\n")
            self.funds -= 1.5 * self.bet
            return True
        else:
            return False

    def print_results(self):
        # clear()
        print("The dealer has a hand " + str(self.dealer_hand) +
              " for a total of " +
              str(self.total(self.dealer_hand)))
        print("You have a hand " + str(self.player_hand) +
              " for a total of " +
              str(self.total(self.player_hand)))

    def score(self, bet):
        self.print_results()
        playerval = self.total(self.player_hand)
        dealerval = self.total(self.dealer_hand)
        if playerval > 21:
            print("Sorry. You busted. You lose.\n")
            self.funds -= bet
        elif dealerval > 21:
            print("Dealer busts. You win!\n")
            self.funds += bet
        elif playerval < dealerval:
            print("Sorry. Your score is lower.\n")
            self.funds -= bet
        elif playerval > dealerval:
            print("Your score is higher. You win\n")
            self.funds += bet
        else:
            print("It's a tie\n")

    def reset(self):
        self.dealer_hand = self.deal()
        self.player_hand = self.deal()
        self.num_plays = 0
        if self.blackjack(): self.reset()

    def current_state(self):
        """
        This will tell you the state of the game
        :return:
        """
        return [self.total(self.dealer_hand), self.total(self.player_hand)]

    def oneStep(self, act):  # act = 0 (stand); 1 (hit); 2 (double)
        # When the cards have been dealt, roundEnd is zero.
        # In between rounds, it is equal to one
        if (act == 0):  # STAND
            while self.total(self.dealer_hand) < 17:
                self.hit(self.dealer_hand)
            self.score(self.bet)
            return self.current_state(), self.funds, self.game_over()

        elif (act == 1):  # HIT
            self.hit(self.player_hand)
            if self.total(self.player_hand) > 21:
                self.score(self.bet)
                return self.current_state(), self.funds, self.game_over()
            else:
                return self.current_state(), self.funds, self.game_over()  # return state and the fact the game isn't over

        else:  # DOUBLE
            self.hit(self.player_hand)
            if self.total(self.player_hand) <= 21:
                while self.total(self.dealer_hand) < 17:
                    self.hit(self.dealer_hand)
            self.score(2 * self.bet)
            return self.current_state(), self.funds, self.game_over()


    def random_game(self):
        num_epochs = 5
        for e in range(num_epochs):
            print("Epoch: {:d}".format(e))
            self.reset()
            game_over = False
            while not game_over:
                action = random.randint(0, 2)
                print("action =", action, " Money =", self.funds)
                game_over, scores = self.oneStep(action)
        print("\nFinal score =", self.funds)



    def interactive_game(self):
        choice = 0
        # clear()
        print("WELCOME TO BLACKJACK!\n")
        self.dealer_hand = self.deal()
        self.player_hand = self.deal()
        while choice != "q":
            print("The dealer is showing a " + str(self.dealer_hand[0]))
            print("You have $", self.funds, "and a hand " + str(self.player_hand) + " for a total of " +
                  str(self.total(self.player_hand)))
            if self.blackjack():
                if self.game_over():
                    break
                else:
                    continue
                # print("You have $", funds)
            choice = input("Do you want to [H]it, [S]tand, [D]doule, or [Q]uit: ").lower()
            # clear()
            if choice == "d":
                self.hit(self.player_hand)
                while self.total(self.dealer_hand) < 17:
                    self.hit(self.dealer_hand)
                    self.score(2 * self.bet)
                    if self.game_over(): break
            elif choice == "h":
                self.hit(self.player_hand)
                if self.total(self.player_hand) > 21:
                    self.score(self.bet)
                    if self.game_over(): break
                else:
                    self.print_results()
            elif choice == "s":
                while self.total(self.dealer_hand) < 17:
                    self.hit(self.dealer_hand)
                self.score(self.bet)
                if self.game_over(): break
            elif choice == "q":
                print("Bye!")
                exit()








if __name__ == "__main__":
    bj = Blackjack()
    bj.interactive_game()
    # bj.random_game()