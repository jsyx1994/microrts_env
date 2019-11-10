package gym;

import ai.abstraction.WorkerRush;
import ai.core.AI;
import ai.*;
import gui.PhysicalGameStatePanel;

import javax.swing.JFrame;

import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;

/**
 * @author santi, Tom
 */

public class GymInterface {
    private static int maxEpisodes = 100000;
    private static int timeBudget = 100;
    private static int maxCycles = 5000;
    private static int period = 20;
    private static long port = 9898;
    private static String map = System.getProperty("user.home") + "/microrts_env/maps/16x16/basesWorkers16x16.xml";


    public static void main(String args[]) throws Exception {
//        for (int i = 0; i < args.length; ++i){
//            System.out.println(args[i]);
//        }
        CliArgs cliArgs = new CliArgs(args);
        port = cliArgs.switchPresent("--port") ? cliArgs.switchLongValue("--port") : port;
        map = cliArgs.switchPresent("--map") ? cliArgs.switchValue("--map") : map;

        System.out.println("Client received info:");
        System.out.println(port);
        System.out.println(map);


        UnitTypeTable utt = new UnitTypeTable();

        PhysicalGameState pgs = PhysicalGameState.load(map, utt);
//        PhysicalGameState pgs = MapGenerator.basesWorkers8x8Obstacle();

        GameState gs = new GameState(pgs, utt);
        boolean gameover = false;

//        AI ai1 = new WorkerRush(utt, new BFSPathFinding());
//        AI ai1 = new GymSocketAI(100,0, "127.0.0.1", 9898, GymSocketAI.LANGUAGE_XML, utt);
        GymSocketAI ai1 = new GymSocketAI(timeBudget, 0, "127.0.0.1", (int) port, GymSocketAI.LANGUAGE_JSON, utt);

        AI ai2 = new WorkerRush(utt);


        ai1.reset(gs, 0);
        ai2.reset();

        JFrame w = PhysicalGameStatePanel.newVisualizer(gs, 640, 640, false, PhysicalGameStatePanel.COLORSCHEME_BLACK);
//        JFrame w = PhysicalGameStatePanel.newVisualizer(gs,640,640,false,PhysicalGameStatePanel.COLORSCHEME_WHITE);

//        ai1.preGameAnalysis(gs, 1000, ".");
//        ai2.preGameAnalysis(gs, 1000, ".");

        long nextTimeToUpdate = System.currentTimeMillis() + period;
        do {
            if (System.currentTimeMillis() >= nextTimeToUpdate) {
                PlayerAction pa1 = ai1.getAction(0, gs);
                PlayerAction pa2 = ai2.getAction(1, gs);
                gs.issueSafe(pa1);
                gs.issueSafe(pa2);

                // simulate:
                gameover = gs.cycle();
                w.repaint();
                nextTimeToUpdate += period;
            } else {
                try {
                    Thread.sleep(1);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        } while (!gameover && gs.getTime() < maxCycles);

        System.out.println("Game Over");
    }
}
