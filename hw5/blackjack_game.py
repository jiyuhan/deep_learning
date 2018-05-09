import sys
import random

###### MAIN GAME FUNCTION BEGINS ######


def mainGame():
    """ Function that contains all the game logic. """

    def gameOver():
        """ Displays a game over screen in its own little loop. It is called when it has been determined that the player's funds have
        run out. All the player can do from this screen is exit the game."""
        print("Game is over, you are out of cash")

    ######## DECK FUNCTIONS BEGIN ########
    def shuffle(deck):
        """ Shuffles the deck using an implementation of the Fisher-Yates shuffling algorithm. n is equal to the length of the
        deck - 1 (because accessing lists starts at 0 instead of 1). While n is greater than 0, a random number k between 0
        and n is generated, and the card in the deck that is represented by the offset n is swapped with the card in the deck
        represented by the offset k. n is then decreased by 1, and the loop continues. """

        n = len(deck) - 1
        while n > 0:
            k = random.randint(0, n)
            deck[k], deck[n] = deck[n], deck[k]
            n -= 1

        return deck

    def createDeck():
        """ Creates a default deck which contains all 52 cards and returns it. """

        deck = ['sj', 'sq', 'sk', 'sa', 'hj', 'hq', 'hk', 'ha',
                'cj', 'cq', 'ck', 'ca', 'dj', 'dq', 'dk', 'da']
        values = range(2, 11)
        for x in values:
            spades = "s" + str(x)
            hearts = "h" + str(x)
            clubs = "c" + str(x)
            diamonds = "d" + str(x)
            deck.append(spades)
            deck.append(hearts)
            deck.append(clubs)
            deck.append(diamonds)
        return deck

    def returnFromDead(deck, deadDeck):
        """ Appends the cards from the deadDeck to the deck that is in play. This is called when the main deck
        has been emptied. """

        for card in deadDeck:
            deck.append(card)
        del deadDeck[:]
        deck = shuffle(deck)

        return deck, deadDeck

    def deckDeal(deck, deadDeck):
        """ Shuffles the deck, takes the top 4 cards off the deck, appends them to the player's and dealer's hands, and
        returns the player's and dealer's hands. """

        deck = shuffle(deck)
        dealerHand, playerHand = [], []

        cardsToDeal = 4

        while cardsToDeal > 0:
            if len(deck) == 0:
                deck, deadDeck = returnFromDead(deck, deadDeck)

            # deal the first card to the player, second to dealer, 3rd to player, 4th to dealer, based on divisibility (it starts at 4, so it's even first)
            if cardsToDeal % 2 == 0:
                playerHand.append(deck[0])
            else:
                dealerHand.append(deck[0])

            del deck[0]
            cardsToDeal -= 1

        return deck, deadDeck, playerHand, dealerHand

    def hit(deck, deadDeck, hand):
        """ Checks to see if the deck is gone, in which case it takes the cards from
        the dead deck (cards that have been played and discarded)
        and shuffles them in. Then if the player is hitting, it gives
        a card to the player, or if the dealer is hitting, gives one to the dealer."""

        # if the deck is empty, shuffle in the dead deck
        if len(deck) == 0:
            deck, deadDeck = returnFromDead(deck, deadDeck)

        hand.append(deck[0])
        del deck[0]

        return deck, deadDeck, hand

    def checkValue(hand):
        """ Checks the value of the cards in the player's or dealer's hand. """

        totalValue = 0

        for card in hand:
            value = card[1:]

            # Jacks, kings and queens are all worth 10, and ace is worth 11
            if value == 'j' or value == 'q' or value == 'k':
                value = 10
            elif value == 'a':
                value = 11
            else:
                value = int(value)

            totalValue += value

        if totalValue > 21:
            for card in hand:
                # If the player would bust and he has an ace in his hand, the ace's value is diminished by 10
                # In situations where there are multiple aces in the hand, this checks to see if the total value
                # would still be over 21 if the second ace wasn't changed to a value of one. If it's under 21, there's no need
                # to change the value of the second ace, so the loop breaks.
                if card[1] == 'a':
                    totalValue -= 10
                if totalValue <= 21:
                    break
                else:
                    continue

        return totalValue

    def blackJack(deck, deadDeck, playerHand, dealerHand, funds, bet, cards, cardSprite):
        """ Called when the player or the dealer is determined to have blackjack. Hands are compared to determine the outcome. """

        playerValue = checkValue(playerHand)
        dealerValue = checkValue(dealerHand)

        if playerValue == 21 and dealerValue == 21:
            # The opposing player ties the original blackjack getter because he also has blackjack
            # No money will be lost, and a new hand will be dealt
            print("Blackjack! The dealer also has blackjack, so it's a push!")
            deck, playerHand, dealerHand, deadDeck, funds, roundEnd = endRound(
                deck, playerHand, dealerHand, deadDeck, funds, 0, bet, cards, cardSprite)

        elif playerValue == 21 and dealerValue != 21:
            # Dealer loses
            print("Blackjack! You won $%.2f." % (bet*1.5))
            deck, playerHand, dealerHand, deadDeck, funds, roundEnd = endRound(
                deck, playerHand, dealerHand, deadDeck, funds, bet, 0, cards, cardSprite)

        elif dealerValue == 21 and playerValue != 21:
            # Player loses, money is lost, and new hand will be dealt
            deck, playerHand, dealerHand, deadDeck, funds, roundEnd = endRound(
                deck, playerHand, dealerHand, deadDeck, funds, 0, bet, cards, cardSprite)
            print("Dealer has blackjack! You lose $%.2f." % (bet))

        return playerHand, dealerHand, deadDeck, funds, roundEnd

    def bust(deck, playerHand, dealerHand, deadDeck, funds, moneyGained, moneyLost, cards, cardSprite):
        """ This is only called when player busts by drawing too many cards. """

        print("You bust! You lost $%.2f." % (moneyLost))

        deck, playerHand, dealerHand, deadDeck, funds, roundEnd = endRound(
            deck, playerHand, dealerHand, deadDeck, funds, moneyGained, moneyLost, cards, cardSprite)

        return deck, playerHand, dealerHand, deadDeck, funds, roundEnd

    def endRound(deck, playerHand, dealerHand, deadDeck, funds, moneyGained, moneyLost, cards, cardSprite):
        """ Called at the end of a round to determine what happens to the cards, the moneyz gained or lost,
        and such. It also shows the dealer's hand to the player, by deleting the old sprites and showing all the cards. """

        if len(playerHand) == 2 and "a" in playerHand[0] or "a" in playerHand[1]:
            # If the player has blackjack, pay his bet back 3:2
            moneyGained += (moneyGained/2.0)

        # Remove old dealer's cards
        cards.empty()

        dCardPos = (50, 70)

        for x in dealerHand:
            card = cardSprite(x, dCardPos)
            dCardPos = (dCardPos[0] + 80, dCardPos[1])
            cards.add(card)

        # Remove the cards from the player's and dealer's hands
        for card in playerHand:
            deadDeck.append(card)
        for card in dealerHand:
            deadDeck.append(card)

        del playerHand[:]
        del dealerHand[:]

        funds += moneyGained
        funds -= moneyLost

        if funds <= 0:
            gameOver()

        roundEnd = 1

        return deck, playerHand, dealerHand, deadDeck, funds, roundEnd

    def compareHands(deck, deadDeck, playerHand, dealerHand, funds, bet, cards, cardSprite):
        """ Called at the end of a round (after the player stands), or at the beginning of a round
        if the player or dealer has blackjack. This function compares the values of the respective hands of
        the player and the dealer and determines who wins the round based on the rules of blacjack. """

        # How much money the player loses or gains, default at 0, changed depending on outcome
        moneyGained = 0
        moneyLost = 0

        dealerValue = checkValue(dealerHand)
        playerValue = checkValue(playerHand)

        # Dealer hits until he has 17 or over
        while 1:
            if dealerValue < 17:
                # dealer hits when he has less than 17, and stands if he has 17 or above
                deck, deadDeck, dealerHand = hit(deck, deadDeck, dealerHand)
                dealerValue = checkValue(dealerHand)
            else:
                # dealer stands
                break

        if playerValue > dealerValue and playerValue <= 21:
            # Player has beaten the dealer, and hasn't busted, therefore WINS
            moneyGained = bet
            deck, playerHand, dealerHand, deadDeck, funds, roundEnd = endRound(
                deck, playerHand, dealerHand, deadDeck, funds, bet, 0, cards, cardSprite)
            print("You won $%.2f." % (bet))
        elif playerValue == dealerValue and playerValue <= 21:
            # Tie
            deck, playerHand, dealerHand, deadDeck, funds, roundEnd = endRound(
                deck, playerHand, dealerHand, deadDeck, funds, 0, 0, cards, cardSprite)
            print("It's a push!")
        elif dealerValue > 21 and playerValue <= 21:
            # Dealer has busted and player hasn't
            deck, playerHand, dealerHand, deadDeck, funds, roundEnd = endRound(
                deck, playerHand, dealerHand, deadDeck, funds, bet, 0, cards, cardSprite)
            print("Dealer busts! You won $%.2f." % (bet))
        else:
            # Dealer wins in every other siutation taht i can think of
            deck, playerHand, dealerHand, deadDeck, funds, roundEnd = endRound(
                deck, playerHand, dealerHand, deadDeck, funds, 0, bet, cards, cardSprite)
            print("Dealer wins! You lost $%.2f." % (bet))

        return deck, deadDeck, roundEnd, funds
    ######## DECK FUNCTIONS END ########
    # cards is the sprite group that will contain sprites for the dealer's cards
    cards = pygame.sprite.Group()
    # playerCards will serve the same purpose, but for the player
    playerCards = pygame.sprite.Group()

    # This creates instances of all the button sprites
    bbU = betButtonUp()
    bbD = betButtonDown()
    standButton = standButton()
    dealButton = dealButton()
    hitButton = hitButton()
    doubleButton = doubleButton()
    
    # The 52 card deck is created
    deck = createDeck()
    # The dead deck will contain cards that have been discarded
    deadDeck = []

    # These are default values that will be changed later, but are required to be declared now
    # so that Python doesn't get confused
    playerHand, dealerHand, dCardPos, pCardPos = [],[],(),()
    mX, mY = 0, 0
    click = 0

    # The default funds start at $100.00, and the initial bet defaults to $10.00
    funds = 100.00
    bet = 10.00

    # This is a counter that counts the number of rounds played in a given session
    handsPlayed = 0

    # When the cards have been dealt, roundEnd is zero.
    #In between rounds, it is equal to one
    roundEnd = 1
    
    # firstTime is a variable that is only used once, to display the initial
    # message at the bottom, then it is set to zero for the duration of the program.
    firstTime = 1
    ###### INITILIZATION ENDS ########
    
    ###### MAIN GAME LOOP BEGINS #######
    while 1:
        
        if bet > funds:
            # If you lost money, and your bet is greater than your funds, make the bet equal to the funds
            bet = funds
        
        if roundEnd == 1 and firstTime == 1:
            # When the player hasn't started. Will only be displayed the first time.
            print("Click on the arrows to declare your bet, then deal to start the game.")
            firstTime = 0
            
        # Show the blurb at the bottom of the screen, how much money left, and current bet    
        print("Funds: $%.2f" %(funds), 1, (255,255,255), (0,0,0))
        print("Bet: $%.2f" %(bet), 1, (255,255,255), (0,0,0))
        print("Round: %i " %(handsPlayed), 1, (255,255,255), (0,0,0))

        switch(sys.stdin.read(1)):
            case 'q': sys.exit()
                break
            case ''

        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    mX, mY = pygame.mouse.get_pos()
                    click = 1
            elif event.type == MOUSEBUTTONUP:
                mX, mY = 0, 0
                click = 0
            
        # Initial check for the value of the player's hand, so that his hand can be displayed and it can be determined
        # if the player busts or has blackjack or not
        if roundEnd == 0:
            # Stuff to do when the game is happening 
            playerValue = checkValue(playerHand)
            dealerValue = checkValue(dealerHand)
    
            if playerValue == 21 and len(playerHand) == 2:
                # If the player gets blackjack
                displayFont, playerHand, dealerHand, deadDeck, funds, roundEnd = blackJack(deck, deadDeck, playerHand, dealerHand, funds,  bet, cards, cardSprite)
                
            if dealerValue == 21 and len(dealerHand) == 2:
                # If the dealer has blackjack
                displayFont, playerHand, dealerHand, deadDeck, funds, roundEnd = blackJack(deck, deadDeck, playerHand, dealerHand, funds,  bet, cards, cardSprite)

            if playerValue > 21:
                # If player busts
                deck, playerHand, dealerHand, deadDeck, funds, roundEnd, displayFont = bust(deck, playerHand, dealerHand, deadDeck, funds, 0, bet, cards, cardSprite)
         
        # Update the buttons 
        # deal 
        deck, deadDeck, playerHand, dealerHand, dCardPos, pCardPos, roundEnd, displayFont, click, handsPlayed = dealButton.update(mX, mY, deck, deadDeck, roundEnd, cardSprite, cards, playerHand, dealerHand, dCardPos, pCardPos, displayFont, playerCards, click, handsPlayed)   
        # hit    
        deck, deadDeck, playerHand, pCardPos, click = hitButton.update(mX, mY, deck, deadDeck, playerHand, playerCards, pCardPos, roundEnd, cardSprite, click)
        # stand    
        deck, deadDeck, roundEnd, funds, playerHand, deadDeck, pCardPos,  displayFont  = standButton.update(mX, mY,   deck, deadDeck, playerHand, dealerHand, cards, pCardPos, roundEnd, cardSprite, funds, bet, displayFont)
        # double
        deck, deadDeck, roundEnd, funds, playerHand, deadDeck, pCardPos, displayFont, bet  = doubleButton.update(mX, mY,   deck, deadDeck, playerHand, dealerHand, playerCards, cards, pCardPos, roundEnd, cardSprite, funds, bet, displayFont)
        # Bet buttons
        bet, click = bbU.update(mX, mY, bet, funds, click, roundEnd)
        bet, click = bbD.update(mX, mY, bet, click, roundEnd)
        # draw them to the screen
        buttons.draw(screen)
         
        # If there are cards on the screen, draw them    
        if len(cards) is not 0:
            playerCards.update()
            playerCards.draw(screen)
            cards.update()
            cards.draw(screen)

        # Updates the contents of the display
        pygame.display.flip()

if __name__ == "__main__":
    mainGame()
